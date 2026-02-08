# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
# Dataset: https://www.cs.cornell.edu/~schnabts/mnar/

import os
import logging
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

try:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window as W
except ImportError:
    pass

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_ISIMPR_COL,
    DEFAULT_IMPRGROUP_COL,
    DEFAULT_PROPENSITY_COL,
)

from recommenders.datasets.download_utils import (
    maybe_download,
    download_path,
    unzip_file,
)

logger = logging.getLogger()

URL_COAT = "https://www.cs.cornell.edu/~schnabts/mnar/coat.zip"


DEFAULT_COAT_PANDAS_TYPE = {
    DEFAULT_USER_COL: "int32",
    DEFAULT_ITEM_COL: "int32",
    DEFAULT_RATING_COL: "int32",
    DEFAULT_ISIMPR_COL: "bool",
    DEFAULT_IMPRGROUP_COL: "int32",
    DEFAULT_PROPENSITY_COL: "float64"
}

# =====================================================
# Download
# =====================================================

def download_coat(dest_path=None):
    """
    Download COAT dataset
    """
    with download_path(dest_path) as path:
        zip_path = maybe_download(url=URL_COAT, work_directory=path)
    return zip_path


# =====================================================
# ASCII loaders
# =====================================================

def _load_ascii_matrix(path):
    """
    Load COAT .ascii matrix files
    Rows = users, columns = items
    Missing entries are marked as 0
    """
    mat = np.loadtxt(path)
    return mat


def _matrix_to_interactions(mat):
    """
    Convert matrix to (user, item, rating)
    """
    rows, cols = np.nonzero(mat)
    df = pd.DataFrame({
        DEFAULT_USER_COL: rows.astype("int32"),
        DEFAULT_ITEM_COL: cols.astype("int32"),
        DEFAULT_RATING_COL: mat[rows, cols].astype("int32"),
        DEFAULT_ISIMPR_COL: True,
        DEFAULT_IMPRGROUP_COL: rows.astype("int32")
    })
    return df


# =====================================================
# Extract + preprocess
# =====================================================

def extract_coat(
    data_zip,
    train_folder="train",
    valid_folder="valid",
    utils_folder="utils",
    clean_zip_file=True,
):
    """
    Extract COAT dataset and create train/test splits
    preserving propensities and features.
    """
    root = os.path.dirname(data_zip)

    tmp_dir = TemporaryDirectory()
    unzip_file(data_zip, tmp_dir.name, clean_zip_file=clean_zip_file)

    # -----------------------------
    # Load matrices
    # -----------------------------
    train_mat = _load_ascii_matrix(os.path.join(tmp_dir.name, "coat", "train.ascii"))
    test_mat = _load_ascii_matrix(os.path.join(tmp_dir.name, "coat", "test.ascii"))
    prop_mat = _load_ascii_matrix(os.path.join(tmp_dir.name, "coat", "propensities.ascii"))

    train_df = _matrix_to_interactions(train_mat)
    test_df = _matrix_to_interactions(test_mat)

    # -----------------------------
    # Propensities
    # -----------------------------
    prop_rows, prop_cols = np.nonzero(prop_mat)
    prop_df = pd.DataFrame({
        DEFAULT_USER_COL: prop_rows.astype("int32"),
        DEFAULT_ITEM_COL: prop_cols.astype("int32"),
        DEFAULT_PROPENSITY_COL: prop_mat[prop_rows, prop_cols].astype("float64"),
    })

    train_df = train_df.merge(prop_df, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
    train_df[DEFAULT_PROPENSITY_COL] = train_df[DEFAULT_PROPENSITY_COL].fillna(0.0)
    test_df = test_df.merge(prop_df, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="left")
    test_df[DEFAULT_PROPENSITY_COL] = test_df[DEFAULT_PROPENSITY_COL].fillna(0.0)

    # -----------------------------
    # Item features (same pattern as kuai.py)
    # -----------------------------
    item_feat = np.loadtxt(
        os.path.join(tmp_dir.name, "coat", "user_item_features", "item_features.ascii")
    )
    item_cols = pd.read_csv(
        os.path.join(tmp_dir.name, "coat", "user_item_features", "item_features_map.txt"),
        header=None
    )[0].tolist()

    item_features = pd.DataFrame(item_feat, columns=item_cols)
    item_features[DEFAULT_ITEM_COL] = item_features.index.astype("int32")

    # -----------------------------
    # User features (utils)
    # -----------------------------
    user_feat = np.loadtxt(
        os.path.join(tmp_dir.name, "coat", "user_item_features", "user_features.ascii")
    )
    user_cols = pd.read_csv(
        os.path.join(tmp_dir.name, "coat", "user_item_features", "user_features_map.txt"),
        header=None
    )[0].tolist()

    user_features = pd.DataFrame(user_feat, columns=user_cols)
    user_features[DEFAULT_USER_COL] = user_features.index.astype("int32")

    # -----------------------------
    # Write output
    # -----------------------------
    train_path = os.path.join(root, train_folder)
    test_path = os.path.join(root, valid_folder)
    utils_path = os.path.join(root, utils_folder)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(utils_path, exist_ok=True)

    train_df.to_csv(os.path.join(train_path, "interactions.csv"), index=False)
    test_df.to_csv(os.path.join(test_path, "interactions.csv"), index=False)

    prop_df.to_csv(os.path.join(train_path, "propensities.csv"), index=False)
    prop_df.to_csv(os.path.join(test_path, "propensities.csv"), index=False)

    item_features.to_csv(os.path.join(train_path, "item_features.csv"), index=False)
    item_features.to_csv(os.path.join(test_path, "item_features.csv"), index=False)
    user_features.to_csv(os.path.join(utils_path, "user_features.csv"), index=False)

    tmp_dir.cleanup()

    return train_path, test_path, utils_path


# =====================================================
# Spark loaders
# =====================================================

def get_df(spark, data_path, topandas=True):
    """
    Load interactions as Spark DataFrame
    """
    w_user = W.partitionBy(DEFAULT_USER_COL)

    df = (
        spark.read.csv(os.path.join(data_path, "interactions.csv"), header=True)
        .select(
            F.col(DEFAULT_USER_COL).cast("int"),
            F.col(DEFAULT_ITEM_COL).cast("int"),
            # Binarize
            F.when(
                # F.col(DEFAULT_RATING_COL).cast("int") > 3, 1
                F.col(DEFAULT_RATING_COL).cast("int") >= F.mean(F.col(DEFAULT_RATING_COL).cast("int")).over(w_user), 1
            ).otherwise(0).alias(DEFAULT_RATING_COL),
            F.col(DEFAULT_ISIMPR_COL).cast("boolean"),
            F.col(DEFAULT_IMPRGROUP_COL).cast("int"),
            F.col(DEFAULT_PROPENSITY_COL).cast("float")
        )
    )

    if topandas:
        df = df.toPandas().astype(DEFAULT_COAT_PANDAS_TYPE)

    return df


def get_features(spark, train, data_path, utils_path, topandas=True):
    """
    Load item and user features for Spark
    """
    item_features = spark.read.csv(
        os.path.join(data_path, "item_features.csv"), header=True
    )

    user_features = spark.read.csv(
        os.path.join(utils_path, "user_features.csv"), header=True
    )

    if topandas:
        item_features = item_features.toPandas().astype("float32").astype({DEFAULT_ITEM_COL: "int32"})
        user_features = user_features.toPandas().astype("float32").astype({DEFAULT_USER_COL: "int32"})

    return user_features, item_features
