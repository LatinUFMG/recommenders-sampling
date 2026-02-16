# Copyright (c)
# Licensed under the MIT License.
# Dataset: Yahoo! Music Webscope R3
# ydata-ymusic-rating-study-v1_0

import os
import tarfile
import logging
from tempfile import TemporaryDirectory

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
    DEFAULT_ISIMPR_COL,
    DEFAULT_IMPRGROUP_COL,
)


DEFAULT_R3_PANDAS_TYPE = {
    DEFAULT_USER_COL: "int32",
    DEFAULT_ITEM_COL: "int32",
    DEFAULT_RATING_COL: "int32",
    DEFAULT_ISIMPR_COL: "bool",
    DEFAULT_IMPRGROUP_COL: "int32",
}

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def extract_r3(
    tgz_path,
    train_folder="train",
    valid_folder="valid",
    utils_folder="utils",
):
    """
    Extract Yahoo! Music R3 dataset from local .tgz archive
    """
    root = os.path.dirname(tgz_path)

    tmp_dir = TemporaryDirectory()
    _untar(tgz_path, tmp_dir.name)

    base = tmp_dir.name

    train_raw = _find_file(base, "train.txt")
    test_raw = _find_file(base, "test.txt")
    survey_q = _find_file(base, "survey-questions.txt")
    survey_a = _find_file(base, "survey-answers.txt")

    # --------------------------------------------------
    # Interactions
    # --------------------------------------------------
    train_df = _load_interactions(train_raw)
    test_df = _load_interactions(test_raw)

    train_path = os.path.join(root, train_folder)
    test_path = os.path.join(root, valid_folder)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_df.to_csv(os.path.join(train_path, "interactions.csv"), index=False)
    test_df.to_csv(os.path.join(test_path, "interactions.csv"), index=False)

    # --------------------------------------------------
    # Utils (survey)
    # --------------------------------------------------
    utils_path = os.path.join(root, utils_folder)
    os.makedirs(utils_path, exist_ok=True)

    _process_survey(survey_q, survey_a, utils_path)

    tmp_dir.cleanup()
    return train_path, test_path, utils_path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _untar(tgz_path, dst):
    """Extract .tgz archive"""
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dst)


def _find_file(root, keyword):
    for path, _, files in os.walk(root):
        for f in files:
            if keyword in f:
                return os.path.join(path, f)
    raise FileNotFoundError(f"File containing '{keyword}' not found")


def _load_interactions(path):
    """
    Load interactions file
    Format: user_id<TAB>item_id<TAB>rating
    """
    df = pd.read_csv(
        path,
        sep="\t",
        names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL],
    )

    df[DEFAULT_USER_COL] = df[DEFAULT_USER_COL].astype("int32")
    df[DEFAULT_ITEM_COL] = df[DEFAULT_ITEM_COL].astype("int32")
    df[DEFAULT_RATING_COL] = df[DEFAULT_RATING_COL].astype("int32")

    # Dummy fields for compatibility
    df[DEFAULT_ISIMPR_COL] = True
    df[DEFAULT_IMPRGROUP_COL] = df[DEFAULT_USER_COL]

    return df


def _process_survey(q_path, a_path, utils_path):
    """
    Process survey questions and answers
    """
    # -------------------------
    # Questions
    # -------------------------
    with open(q_path, "r") as f:
        questions = f.read()

    with open(os.path.join(utils_path, "survey_questions.txt"), "w") as f:
        f.write(questions)

    # -------------------------
    # Answers
    # -------------------------
    answers = pd.read_csv(
        a_path,
        sep="\t",
        header=None,
    )

    answers.columns = [f"q{i+1}" for i in range(answers.shape[1])]
    answers[DEFAULT_USER_COL] = answers.index.astype("int32") + 1

    answers.to_csv(
        os.path.join(utils_path, "survey_answers.csv"),
        index=False,
    )

    # -------------------------
    # User survey features
    # -------------------------
    survey_feat = answers.copy()
    survey_feat.to_csv(
        os.path.join(utils_path, "user_survey_features.csv"),
        index=False,
    )


# ----------------------------------------------------------------------
# Spark loader
# ----------------------------------------------------------------------

def get_df(spark, data_path: str, topandas=True):
    """
    Load R3 interactions as Spark DataFrame
    """
    w_user = W.partitionBy(DEFAULT_USER_COL)

    df = spark.read.csv(
        os.path.join(data_path, "interactions.csv"),
        header=True,
    ).select(
        F.col(DEFAULT_IMPRGROUP_COL).cast("int"),
        F.col(DEFAULT_USER_COL).cast("int"),
        F.col(DEFAULT_ITEM_COL).cast("int"),
        # F.col(DEFAULT_RATING_COL).cast("int").alias(DEFAULT_RATING_COL + "_nobin"),
        # Binarize
        F.when(
            # F.col(DEFAULT_RATING_COL).cast("int") > 3, 1
            F.col(DEFAULT_RATING_COL).cast("int") >= F.mean(F.col(DEFAULT_RATING_COL).cast("int")).over(w_user), 1
        ).otherwise(0).alias(DEFAULT_RATING_COL),
        F.col(DEFAULT_ISIMPR_COL).cast("boolean"),
    )

    if topandas:
        df = df.toPandas().astype(DEFAULT_R3_PANDAS_TYPE)

    return df
