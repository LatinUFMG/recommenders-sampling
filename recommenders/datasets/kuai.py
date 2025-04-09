# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
# Dataset: https://kuairec.com/

import os
import random
import logging
import shutil
from tempfile import TemporaryDirectory

import pandas as pd

try:
    from pyspark.sql import types as T
    from pyspark.sql import functions as F
    from pyspark.sql import window as W
except ImportError:
    pass  # so the environment without spark doesn"t break

from recommenders.utils.constants import (
    DEFAULT_GENRE_COL,
    DEFAULT_HEADER,
    DEFAULT_IMPRGROUP_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_ISIMPR_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)
from recommenders.datasets.download_utils import (
    maybe_download,
    download_path,
    unzip_file,
)

URL_KUAI = {
    "rec": "https://nas.chongminggao.top:4430/datasets/KuaiRec.zip",
    # TODO: Implement other datasets from Kuai (e.g.: KuaiRand-1K)
    "rand1k": "",
}

DEFAULT_KUAI_PANDAS_TYPE = {
    DEFAULT_IMPRGROUP_COL: "int32",
    DEFAULT_ISIMPR_COL: "bool",
    DEFAULT_USER_COL: "int32",
    DEFAULT_ITEM_COL: "int32",
    DEFAULT_RATING_COL: "int32",
    DEFAULT_TIMESTAMP_COL: "str",
}

USER_FEATURES_KUAI_PANDAS_TYPE = {
    DEFAULT_USER_COL: "int32", "is_lowactive_period": "int32", "is_live_streamer": "int32",
    "is_video_author": "int32", "follow_user_num": "float64", "fans_user_num": "float64",
    "fans_user_num_range": "category", "friend_user_num": "int32", "register_days": "float64",
    "user_active_degree": "category", "friend_user_num_range": "category",
    "follow_user_num_range": "category", "register_days_range": "category",
    "onehot_feat0": "int32", "onehot_feat1": "int32", "onehot_feat2": "int32", "onehot_feat3": "int32",
    "onehot_feat4": "int32", "onehot_feat5": "int32", "onehot_feat6": "int32", "onehot_feat7": "int32",
    "onehot_feat8": "int32", "onehot_feat9": "int32", "onehot_feat10": "int32", "onehot_feat11": "int32",
    "onehot_feat12": "int32", "onehot_feat13": "int32", "onehot_feat14": "int32", "onehot_feat15": "int32",
    "onehot_feat16": "int32", "onehot_feat17": "int32",
}

# ITEM_FEATURES_KUAI_PANDAS_TYPE = {
#     "visible_status": "category", "video_duration": "float64",
#     "author_id": "int32", "video_type": "category", "upload_type": "category", "video_width": "float64",
#     "video_height": "float64", "music_id": "int64", "video_tag_id": "int32", "video_tag_name": "category",
#     "show_cnt": "float64", "show_user_num": "float64", "play_cnt": "float64", "play_user_num": "float64",
#     "play_duration": "float64", "complete_play_cnt": "float64", "complete_play_user_num": "float64",
#     "valid_play_cnt": "float64", "valid_play_user_num": "float64", "long_time_play_cnt": "float64",
#     "long_time_play_user_num": "float64", "short_time_play_cnt": "float64", "short_time_play_user_num": "float64",
#     "play_progress": "float64", "comment_stay_duration": "float64", "like_cnt": "float64",
#     "like_user_num": "float64", "click_like_cnt": "float64", "double_click_cnt": "float64",
#     "cancel_like_cnt": "float64", "cancel_like_user_num": "float64", "comment_cnt": "float64",
#     "comment_user_num": "float64", "direct_comment_cnt": "float64", "reply_comment_cnt": "float64",
#     "delete_comment_cnt": "float64", "delete_comment_user_num": "float64", "comment_like_cnt": "float64",
#     "comment_like_user_num": "float64", "follow_cnt": "float64", "follow_user_num": "float64",
#     "cancel_follow_cnt": "float64", "cancel_follow_user_num": "float64", "share_cnt": "float64",
#     "share_user_num": "float64", "download_cnt": "float64", "download_user_num": "float64", "report_cnt": "float64",
#     "report_user_num": "float64", "reduce_similar_cnt": "float64", "reduce_similar_user_num": "float64",
#     "collect_cnt": "float64", "collect_user_num": "float64", "cancel_collect_cnt": "float64",
#     "cancel_collect_user_num": "float64",
#     # "upload_dt": "str",
# }

ITEM_FEATURES_KUAI_PANDAS_TYPE = {
    DEFAULT_ITEM_COL: "int32",
    # "music_id": "int64", "author_id": "int32",
    # "video_duration": "float64", "video_width": "float64", "video_height": "float64",
    # "video_type": "category", "upload_type": "category",
    "show_cnt": "float64", "play_cnt": "float64", "play_duration": "float64",
    "complete_play_cnt": "float64", "valid_play_cnt": "float64",
    "like_cnt": "float64", "comment_cnt": "float64", "follow_cnt": "float64",
    "share_cnt": "float64", "download_cnt": "float64", "collect_cnt": "float64",
}


logger = logging.getLogger()


def download_kuai(source="rec", dest_path=None):
    """Download Kuai dataset

    Args:
        source (str): Dataset source. One of ["rec", ...]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path

    Returns:
        str: Path to train set. Further splited into train/validaiton.
    """
    if source not in URL_KUAI:
        raise ValueError(f"Wrong size option, available options are {list(URL_KUAI.keys())}")
    url_data = URL_KUAI[source]
    with download_path(dest_path) as path:
        train_path = maybe_download(url=url_data, work_directory=path)
    return train_path


def extract_kuai(data_zip, train_folder="train", valid_folder="valid", utils_folder="utils", clean_zip_file=True):
    """Extract Kuai dataset and split into train/validation.

    Args:
        data_zip (str): Path to train zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set
        utils_folder (str):  Destination forder for utils
        clean_zip_file(str): If zip file must be deleted at the end.

    Returns:
        str, str: Train and validation folders
    """
    root_folder = os.path.dirname(data_zip)

    tmp_dir = TemporaryDirectory()
    unzip_file(data_zip, tmp_dir.name, clean_zip_file=clean_zip_file)

    big_matrix_path = os.path.join(tmp_dir.name, "KuaiRec 2.0/data/big_matrix.csv")
    big_df = pd.read_csv(big_matrix_path)
    # date column has empty period and even using time, there are days with only few interactions
    big_df["date"] = big_df["time"].str.replace("-", "").fillna("0").str.slice(0, 8).astype("int")

    small_matrix_path = os.path.join(tmp_dir.name, "KuaiRec 2.0/data/small_matrix.csv")
    small_df = pd.read_csv(small_matrix_path)
    small_df["date"] = small_df["time"].str.replace("-", "").fillna("0").str.slice(0, 8).astype("int")

    item_daily_features_path = os.path.join(tmp_dir.name, "KuaiRec 2.0/data/item_daily_features.csv")
    daily_df = pd.read_csv(item_daily_features_path)

    dt_split = 20200831

    ### Training set ###
    # train_df = pd.concat([
    #     big_df.query(f"date <= {dt_split} and date > 0"),
    #     small_df.query(f"date <= {dt_split} and date > 0")
    # ]).sort_values("time")
    train_df = big_df.sort_values("time")
    train_daily_df = daily_df#.query(f"date <= {dt_split} and date > 0")

    train_path = os.path.join(root_folder, train_folder)
    os.makedirs(train_path, exist_ok=True)
    train_df.to_csv(os.path.join(train_path, "interactions.csv"), index=False)
    train_daily_df.to_csv(os.path.join(train_path, "item_daily_features.csv"), index=False)

    ### Validation set  ###

    # valid_df = small_df.query(f"date > {dt_split}")
    valid_df = small_df.sort_values("time")
    valid_daily_df = daily_df#.query(f"date > {dt_split}")

    valid_path = os.path.join(root_folder, valid_folder)
    os.makedirs(valid_path, exist_ok=True)
    valid_df.to_csv(os.path.join(valid_path, "interactions.csv"), index=False)
    valid_daily_df.to_csv(os.path.join(valid_path, "item_daily_features.csv"), index=False)

    ### Utils files ###
    utils_path = os.path.join(root_folder, utils_folder)
    os.makedirs(utils_path, exist_ok=True)
    files_cp = ["item_categories.csv", "kuairec_caption_category.csv", "social_network.csv", "user_features.csv"]
    for filename in files_cp:
        shutil.copy(
            os.path.join(tmp_dir.name, f"KuaiRec 2.0/data/{filename}"),
            os.path.join(utils_path, filename)
        )

    tmp_dir.cleanup()
    return train_path, valid_path, utils_path


def get_spark_df(spark, data_path: str):
    """Return KuaiRec Dataset as a Spark Dataframe with specified rows

    Args:
        spark (SparkSession): spark session to load the dataframe into.
        data_path (str): path to dataset.

    Returns:
        pyspark.sql.DataFrame: a mock dataset
    """

    df = spark.read.csv(os.path.join(data_path, "interactions.csv"), header=True)

    # The “watch_ratio” can be deemed as the label of the interaction.
    # Note: there is no “like” signal for this dataset.
    # If you need this binary signal in your scenarios, you can create it yourself.
    # E.g., like = 1 if watch_ratio > 2.0.
    return df.select(
        F.col("user_id").cast("int").alias(DEFAULT_IMPRGROUP_COL),
        F.col("time").alias(DEFAULT_TIMESTAMP_COL),
        F.col("user_id").cast("int").alias(DEFAULT_USER_COL),
        F.col("video_id").cast("int").alias(DEFAULT_ITEM_COL),
        F.when(F.col("watch_ratio") > 2.0, F.lit(1)).otherwise(F.lit(0)).alias(DEFAULT_RATING_COL),
        F.lit(True).alias(DEFAULT_ISIMPR_COL)
    )


def enrich_spark_df(spark, df, data_path: str, utils_path:str):
    ids_columns = [
        # Item Daily Features
        "author_id", "music_id", "video_tag_id",
        "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3", "onehot_feat4", "onehot_feat5",
        "onehot_feat6", "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10",
        "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15",
        "onehot_feat16", "onehot_feat17",
    ]

    categorical_features = [
        # Item Daily Features
        "video_type", "upload_type", "visible_status", "video_tag_name",
        # User Features
        "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
        "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"
    ]

    numeric_features = [
        # Item Daily Features
        "video_duration", "video_width", "video_height", "show_cnt", "show_user_num", "play_cnt",
        "play_user_num", "play_duration", "complete_play_cnt", "complete_play_user_num",
        "valid_play_cnt", "valid_play_user_num", "long_time_play_cnt", "long_time_play_user_num",
        "short_time_play_cnt", "short_time_play_user_num", "play_progress", "comment_stay_duration",
        "like_cnt", "like_user_num", "click_like_cnt", "double_click_cnt", "cancel_like_cnt",
        "cancel_like_user_num", "comment_cnt", "comment_user_num", "direct_comment_cnt",
        "reply_comment_cnt", "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
        "comment_like_user_num", "follow_cnt", "follow_user_num", "cancel_follow_cnt",
        "cancel_follow_user_num", "share_cnt", "share_user_num", "download_cnt", "download_user_num",
        "report_cnt", "report_user_num", "reduce_similar_cnt", "reduce_similar_user_num", "collect_cnt",
        "collect_user_num", "cancel_collect_cnt", "cancel_collect_user_num",
        # User Features
        "follow_user_num", "fans_user_num", "friend_user_num", "register_days",
    ]

    daily_features = spark.read.csv(os.path.join(data_path, "item_daily_features.csv"), header=True)
    user_features = spark.read.csv(os.path.join(utils_path, "user_features.csv"), header=True)

    df_enriched = (
        df
        .join(
            daily_features.withColumnRenamed("video_id", DEFAULT_ITEM_COL),
            on=[DEFAULT_ITEM_COL], how="left"
        )
        .withColumn(
            "datediff",
            F.date_diff(
                F.to_date(F.col("date"), "yyyyMMdd"),
                F.to_date(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSS")
            )
        )
        .where(F.col("datediff") == 1)
        .join(
            user_features
            .withColumnRenamed("user_id", DEFAULT_USER_COL)
            .withColumnRenamed("follow_user_num", "user_follow_user_num"),
            on=[DEFAULT_USER_COL], how="left"
        )
    ).select(list(df.columns) + categorical_features + numeric_features)

    return df_enriched, ids_columns, categorical_features, numeric_features


def get_spark_features(spark, train, data_path: str, utils_path: str):
    # Load daily item features and user features
    daily_features = spark.read.csv(os.path.join(data_path, "item_daily_features.csv"), header=True)
    user_features = spark.read.csv(os.path.join(utils_path, "user_features.csv"), header=True)

    # Select the most recent daily item features by removing duplicates based on "itemID"
    # Assumes there"s a "timestamp" column in daily_features that indicates recency
    max_train_date = train.select(F.max(F.to_date(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSS"))).collect()[0][0]
    daily_features = (
        daily_features
            .select(F.col("video_id").cast("int").alias(DEFAULT_ITEM_COL), "*")
            .withColumn("date_parsed", F.to_date(F.col("date"), "yyyyMMdd"))
            .filter(F.col("date_parsed") <= max_train_date)
            .withColumn("row_num", F.row_number().over(W.Window.partitionBy("itemID").orderBy(F.desc("date_parsed"))))
            .filter(F.col("row_num") == 1)
            .drop("video_id").drop("date_parsed").drop("date").drop("row_num")
        )

    # Remove duplicates from user features based on "userID" (assuming there are no daily features for users)
    user_features = user_features.select(F.col("user_id").cast("int").alias(DEFAULT_USER_COL), "*").drop("user_id")

    return user_features, daily_features
