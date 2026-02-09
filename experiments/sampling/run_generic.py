# %%
import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import gc
import importlib
import json
import os
import psutil
import ray
import ray.train
import ray.tune
import scipy
import sys
import numpy as np
import pandas as pd
import torch

from ray.internal import free as rayfree
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import KFold

# Recommenders dependencies
from recommenders.evaluation.pyrank_evaluation import Math, run_eval_as_ranking, Samplers
from recommenders.models.lightfm import LightFMFModel, LightFMModel
from recommenders.models.implicit import ImplicitModel
from recommenders.models.pop import ClickPOP
from recommenders.models.rand import RandomModel
from recommenders.models.sampling_adapters import SASRecAdapter, LightGCNAdapter, FastAIAdapter,EASEAdapter
from recommenders.models.sar import SAR
from recommenders.utils.constants import (
    DEFAULT_IMPRGROUP_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_ISIMPR_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)
from recommenders.utils.spark_utils import start_or_get_spark

# Constants
MAX_WORKERS = int(os.cpu_count() / 2)
MAX_TUNING = 128
TUNING_METRIC = "mean_mrr"
SAMPLING_TRAIN_TYPE = "full"
DT_FORMAT = "%Y-%m-%d %H:%M"

# Standard Header
HEADER = {
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
    "col_prediction": DEFAULT_PREDICTION_COL,
}

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# %% Utility Functions

def auto_garbage_collect(pct=80.0):
    """
    Call the garbage collection if memory used is greater than 80% of total available memory.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

def get_dataset_module(dataset_name):
    """
    Dynamically imports the dataset module from recommenders.datasets.
    Example: if dataset_name is 'kuai', imports 'recommenders.datasets.kuai'.
    """
    try:
        module = importlib.import_module(f"recommenders.datasets.{dataset_name.lower()}")
        return module
    except ImportError:
        raise ImportError(f"Could not import module 'recommenders.datasets.{dataset_name.lower()}'. "
                          "Ensure the dataset module exists and follows the package structure.")

# %% Core Logic
def objective(config, train, alg_class, fixed_params, user_features, item_features):
    """
    Ray Tune objective function using Global Random K-Fold cross-validation.
    Ensures users and items appear in both train and test sets (randomly).
    """
    # 1. Shuffle
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = []

    # 2. Split
    for train_index, val_index in kf.split(train):

        train_fold_raw = train.iloc[train_index]
        val_fold_raw = train.iloc[val_index]

        # Train fold
        groupby_columns = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        if alg_class == SASRecAdapter:
            groupby_columns.append(DEFAULT_TIMESTAMP_COL)
        train_fold = (
            train_fold_raw
            .groupby(groupby_columns)
            .agg({DEFAULT_RATING_COL: "sum"})
            .reset_index()
        )

        # Validation fold:
        val_fold = val_fold_raw # .query(DEFAULT_ISIMPR_COL)

        all_params = {**fixed_params, **config}
        model = alg_class(**all_params)

        if hasattr(model, "fitf") and (user_features is not None or item_features is not None):
            model.fitf(train_fold, user_features, item_features)
        else:
            model.fit(train_fold)

        result = run_eval_as_ranking(
            model, train_fold, val_fold, metrics=[TUNING_METRIC],
            sampler_type="cncrandom", sample_size=100
        )
        metrics.append(result[TUNING_METRIC])

    avg_metric = np.mean(metrics)
    return {TUNING_METRIC: avg_metric}


def tune_and_load_models(models, train, user_features, item_features, model_base_path):
    """
    Iterates through models, runs Ray Tune to find best params, and returns the model definitions
    updated with the best configuration.
    """
    final_models = []

    for alg_name, alg_class, alg_params in models:
        train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
        model_train_path = os.path.join(model_base_path, train_key)
        os.makedirs(model_train_path, exist_ok=True)

        tuning_name = f"{alg_name}_tuning"
        tuning_path = os.path.join(model_train_path, tuning_name)
        best_params_path = os.path.join(tuning_path, "final_best_params.json")
        best_params = {}

        # Run Tuning if supported
        if hasattr(alg_class, "get_raytune_space"):
            if not os.path.exists(best_params_path):
                print(f"Tuning {alg_name}...")

                # Check if we can restore specific tune run
                abs_tuning_path = os.path.abspath(tuning_path)
                tuner = None

                if ray.tune.Tuner.can_restore(abs_tuning_path):
                    # We need to recreate the objective with parameters to restore
                    objective_fn = ray.tune.with_parameters(
                        ray.tune.with_resources(objective, {"cpu": 4, "gpu": 0.125}),
                        train=train, alg_class=alg_class, fixed_params=alg_params,
                        user_features=user_features, item_features=item_features,
                    )
                    tuner = ray.tune.Tuner.restore(abs_tuning_path, objective_fn)
                else:
                    objective_fn = ray.tune.with_parameters(
                        ray.tune.with_resources(objective, {"cpu": 4, "gpu": 0.125}),
                        train=train, alg_class=alg_class, fixed_params=alg_params,
                        user_features=user_features, item_features=item_features,
                    )
                    algo = HyperOptSearch(metric=TUNING_METRIC, mode="max", random_state_seed=42)
                    tuner = ray.tune.Tuner(
                        objective_fn,
                        tune_config=ray.tune.TuneConfig(
                            mode="max", metric=TUNING_METRIC, search_alg=algo, num_samples=MAX_TUNING,
                        ),
                        run_config=ray.train.RunConfig(
                            name=tuning_name, storage_path=os.path.abspath(model_train_path),
                        ),
                        param_space=alg_class.get_raytune_space(alg_name),
                    )

                tuning_results = tuner.fit()
                best_params = tuning_results.get_best_result().config

                with open(best_params_path, "w") as writer:
                    json.dump(best_params, writer)
            else:
                print(f"Loading best params for {alg_name}...")
                with open(best_params_path, "r") as reader:
                    best_params = json.load(reader)

        final_models.append((alg_name, alg_class, {**alg_params, **best_params}))

    return final_models

# %% Remote Evaluation

def run_predict(
    sampling_exp_type, sample_size, model_train_path, alg_name, alg_class, alg_params,
    train, test, result_key, results_file,
):
    METRICS = [
        "group_auc", "mean_mrr",
        "ndcg@5;10;50;100", "precision@5;10;50;100", "recall@5;10;50;100",
        "unbiased_ndcg@5;10;50;100", "unbiased_precision@5;10;50;100", "unbiased_recall@5;10;50;100",
        "mean_mrr@5;10;50;100"
    ]

    model_to_eval = alg_class(**alg_params)
    if not model_to_eval.load_model(model_train_path):
        raise FileNotFoundError(f"Error to load model {alg_class} from {model_train_path}")

    print(f"Start: {alg_name} | {sampling_exp_type.name} | Size: {sample_size}")

    test_with_preds = run_eval_as_ranking(
        model_to_eval, train, test.copy(), METRICS, sampling_exp_type, sample_size,
        byline=True, returndf=True,
    )
    test_with_preds["result_key"] = result_key

    test_with_preds.to_parquet(results_file)
    del test_with_preds
    auto_garbage_collect()

    return True

remote_run_predict = ray.remote(run_predict)


# %% Main Execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='GeneralizedSamplingExperiments',
        description='Execute experiments for sampling evaluation using dataset modules'
    )
    parser.add_argument(
        '--dataset', required=True, type=str,
        help="Name of the dataset module (e.g., 'kuai', 'criteo') available in recommenders.datasets"
    )
    parser.add_argument(
        '-d', '--dt', default="",
        help=f"Execution date in format {DT_FORMAT.replace('%', '')} (default: now)"
    )
    parser.add_argument(
        "--ignore_coverage_filter", action="store_false", default=True,
        help="Don't apply the filter to keep items with 100%% user coverage (default: %(default)s)"
    )

    args = parser.parse_args()

    # 1. Load Dataset Module Dynamically
    dataset_name = args.dataset
    dataset_module = get_dataset_module(dataset_name)

    # 2. Start Spark
    spark = start_or_get_spark(dataset_name, memory="16g")
    spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

    # 3. Setup Paths and Data
    data_path = f"./data/{dataset_name}"
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    utils_path = os.path.join(data_path, "utils")

    # Check/Download Data
    # Assumes module has download_<name> and extract_<name> functions
    if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
        print(f"Downloading {dataset_name} data...")
        download_fn = getattr(dataset_module, f"download_{dataset_name}")
        extract_fn = getattr(dataset_module, f"extract_{dataset_name}")

        # We assume download function takes a 'source' param (like 'rec' in kuai) or just path
        # Adjusting to generic call, falling back to just path if source logic is specific to Kuai
        try:
            data_zip = download_fn("rec", data_path) # Try 'rec' as default source
        except TypeError:
            data_zip = download_fn(data_path)

        extract_fn(data_zip, clean_zip_file=False)

    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"{valid_path} not found.")

    # 4. Setup Directories
    exec_dt = datetime.datetime.now().strftime(DT_FORMAT)
    try:
        if args.dt:
            exec_dt = datetime.datetime.strptime(args.dt, DT_FORMAT).strftime(DT_FORMAT)
            print(f"Continue execution with {exec_dt} prefix.")
    except ValueError:
        print(f"Date parsing failed. Starting new execution with {exec_dt}.")

    results_path = os.path.join(data_path, "result", exec_dt)
    model_path = os.path.join(data_path, "model")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 5. Load Dataframes
    print("Loading Train/Test...")
    train = dataset_module.get_df(spark, train_path, topandas=True)
    test = dataset_module.get_df(spark, valid_path, topandas=True)

    # 6. Apply Filter (Coverage)
    print("Size test before filter", test.shape)
    if args.ignore_coverage_filter:
        total_users = test[DEFAULT_USER_COL].nunique()
        percentages = test.groupby(DEFAULT_ITEM_COL)[DEFAULT_USER_COL].nunique().reset_index()
        percentages.columns = [DEFAULT_ITEM_COL, "unique_users"]
        percentages["percent"] = (percentages["unique_users"] / total_users) * 100
        video_ids_to_keep = percentages.query("percent == 100")[DEFAULT_ITEM_COL].values
        test = test.query(f"{DEFAULT_ITEM_COL} in @video_ids_to_keep").copy().reset_index(drop=True)
    print("Size test after filter", test.shape)

    # 7. Load Features
    print("Loading Features...")
    user_features = None
    item_features = None
    if hasattr(dataset_module, "get_features"):
        user_features, item_features = dataset_module.get_features(
            spark, dataset_module.get_df(spark, train_path, topandas=True), train_path, utils_path, topandas=True
        )

    # Dynamic Feature Type Casting
    user_feat_types = getattr(dataset_module, f"USER_FEATURES_{dataset_name.upper()}_PANDAS_TYPE", {})
    item_feat_types = getattr(dataset_module, f"ITEM_FEATURES_{dataset_name.upper()}_PANDAS_TYPE", {})

    if user_feat_types:
        user_features[list(user_feat_types.keys())] = (
            user_features[list(user_feat_types.keys())].fillna(0).astype(user_feat_types)
        )

    if item_feat_types:
        item_features[list(item_feat_types.keys())] = (
            item_features[list(item_feat_types.keys())].fillna(0).astype(item_feat_types)
        )
        item_features = item_features[list(item_feat_types.keys())]

    spark.stop()

    # 8. Define Models
    sar_params = {
        "similarity_type": "jaccard",
        "time_decay_coefficient": 0,
        "time_now": None,
        "timedecay_formula": False,
    }

    MODELS = [
        ("sar-jaccard", SAR, {**sar_params, **HEADER}),
        ("sar-cosine", SAR, {**sar_params, **{"similarity_type": "cosine"}, **HEADER}),
        ("clickpop", ClickPOP, HEADER),
        ("random", RandomModel, HEADER),
        ("als", ImplicitModel, HEADER),
        ("bpr", ImplicitModel, HEADER),
        ("lightfm", LightFMModel, HEADER),  # no fatures
        ("lightfmf", LightFMFModel, HEADER),  # with features
        ("sasrec", SASRecAdapter, HEADER),
        ("lightgcn", LightGCNAdapter, HEADER),
        ("fastai", FastAIAdapter, HEADER),
        ("ease", EASEAdapter, HEADER),
    ]

    # 9. Ray Initialization
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    # 10. Tune and Train
    tuned_models = tune_and_load_models(MODELS, train, user_features, item_features, model_path)

    # 11. Final Training (Full Train + Test Clicks)
    train_fold_with_time = (
        # Adds historic of clicks from users only in test set (validation (impressions) still preserved)
        pd.concat([train, test.query(f"~{DEFAULT_ISIMPR_COL}")])
        .groupby([DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        .agg({DEFAULT_RATING_COL: "sum"})
        .reset_index()
    )

    train_fold_agg = (
        pd.concat([train, test.query(f"~{DEFAULT_ISIMPR_COL}")])
        .groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        .agg({DEFAULT_RATING_COL: "sum"})
        .reset_index()
    )

    print("Executing Final Training...")
    for alg_name, alg_class, alg_params in tuned_models:
        torch.cuda.empty_cache()
        train_fold_full = train_fold_agg if alg_class != SASRecAdapter else train_fold_with_time
        train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
        model_train_path = os.path.join(model_path, train_key)
        os.makedirs(model_train_path, exist_ok=True)

        model = alg_class(**alg_params)
        if not model.load_model(model_train_path):
            if hasattr(model, "fitf") and (user_features is not None or item_features is not None):
                model.fitf(train_fold_full, user_features, item_features)
            else:
                model.fit(train_fold_full)
            model.save_model(model_train_path)

    # 12. Distributed Evaluation
    RANDOM_VARIATIONS = [0, 1, 2, 5, 10, 20, 50, 100, 200] #, 500, 1000]
    futures = []

    train_rayid = ray.put(train)
    test_rayid = ray.put(test)

    print("Submitting Evaluation Tasks...")
    for alg_name, alg_class, alg_params in tuned_models:
        train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
        model_train_path = os.path.join(model_path, train_key)

        for sampling_exp_type in Samplers:
            variations = [None]
            if sampling_exp_type.name.lower().endswith("random"):
                variations = RANDOM_VARIATIONS
            elif sampling_exp_type in [Samplers.POPULARITY, Samplers.POSITIVITY, Samplers.SKEW, Samplers.WTD, Samplers.WTDH]:
                variations = RANDOM_VARIATIONS[1:]

            for sample_size in variations:
                exp_key = f"{alg_name}_{sampling_exp_type.name}"

                result_key = f"train:{train_key}-test_{dataset_name}:{exp_key}"
                if sample_size is not None:
                    result_key += f"@{sample_size}"

                results_file = os.path.join(results_path, result_key + ".parquet")

                if os.path.exists(results_file):
                    continue

                # Enable sequentially running
                # run_predict(
                #     Samplers.CNCRANDOM, 0, model_train_path,
                #     alg_name, alg_class, alg_params,
                #     train_fold_full, test,
                #     result_key, results_file
                # )
                # 0/0

                future = remote_run_predict.options(num_cpus=4, num_gpus=0.125).remote(
                    sampling_exp_type, sample_size, model_train_path,
                    alg_name, alg_class, alg_params,
                    train_rayid,
                    test_rayid,
                    result_key, results_file
                )
                futures.append(future)

    rayfree([test_rayid])

    # 13. Collect Results
    # Wait for all the tasks to complete and get results
    print("Waiting futures...")
    results = ray.get(futures)

    # 14. Merge Results
    print("Merging Files...")
    results_to_merge = []
    final_results_file = os.path.join(results_path, exec_dt + "_" + dataset_name + ".parquet")

    for file_to_merge in os.listdir(results_path):
        if file_to_merge.endswith(".parquet") and not file_to_merge.startswith(exec_dt):
             try:
                df = pd.read_parquet(os.path.join(results_path, file_to_merge))
                results_to_merge.append(df)
             except Exception:
                 pass

    if results_to_merge:
        pd.concat(results_to_merge).to_parquet(final_results_file)
        print(f"Saved merged results to {final_results_file}")

    ray.shutdown()
