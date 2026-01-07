# %%
import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import gc
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

from ray.internal import free as rayfree
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import KFold

from recommenders.datasets import kuai
from recommenders.evaluation.pyrank_evaluation import Math, run_eval_as_ranking, Samplers
from recommenders.models.lightfm import LightFMFModel, LightFMModel, LightFMFUModel
from recommenders.models.implicit import ImplicitModel
from recommenders.models.pop import ClickPOP
from recommenders.models.rand import RandomModel
from recommenders.models.sar import SAR
from recommenders.utils.constants import (
    DEFAULT_GENRE_COL,
    DEFAULT_HEADER,
    DEFAULT_IMPRGROUP_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_ISIMPR_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)
from recommenders.utils.spark_utils import start_or_get_spark


MAX_WORKERS = int(os.cpu_count() / 2)
MAX_TUNING = 128
TUNING_METRIC = "mean_mrr"
SAMPLING_TRAIN_TYPE = "full"


print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# %%
spark = start_or_get_spark("kuai", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

# %%
# Select Kuai data source. Options: rec
kuai_source = "rec"

# %%
# Download and load data
data_path = f"./data/Kuai{kuai_source.capitalize()}"

train_path = os.path.join(data_path, "train")
if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
    data_zip = kuai.download_kuai(kuai_source, data_path)
    kuai.extract_kuai(data_zip, clean_zip_file=False)

valid_path = os.path.join(data_path, "valid")
if not os.path.exists(valid_path) or len(os.listdir(valid_path)) == 0:
    raise FileNotFoundError(f"{valid_path} not found.")

utils_path = os.path.join(data_path, "utils")
if not os.path.exists(utils_path) or len(os.listdir(utils_path)) == 0:
    raise FileNotFoundError(f"{utils_path} not found.")

# %%
model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

# %%
DT_FORMAT = "%Y-%m-%d %H:%M"
arg_parser = argparse.ArgumentParser(
    prog='SamplingExperiments',
    description='Execute experiments for sampling evaluation',
    epilog='...'
)
arg_parser.add_argument(
    '-d', '--dt', default="",
    help=f"""
        Execution date in format {DT_FORMAT.replace('%', '')}
        to continue experiment from last execution (default: %(default)s)
    """
)
arg_parser.add_argument(
    "--ignore_coverage_filter", action="store_false", default=True,
    help="Don't apply the filter to keep items with 100\% user coverage (default: %(default)s)"
)
args = arg_parser.parse_args()

# %%
exec_dt = datetime.datetime.now().strftime(DT_FORMAT)
try:
    exec_dt = datetime.datetime.strptime(args.dt, DT_FORMAT).strftime(DT_FORMAT)
    print(f"Continue execution with {exec_dt} prefix.")
except ValueError:
    print(f"Starting new execution with {exec_dt} prefix.")

results_path = os.path.join(data_path, "result", exec_dt)
os.makedirs(results_path, exist_ok=True)

# %%
train = kuai.get_spark_df(spark, train_path).toPandas().astype(kuai.DEFAULT_KUAI_PANDAS_TYPE)
test = kuai.get_spark_df(spark, valid_path).toPandas().astype(kuai.DEFAULT_KUAI_PANDAS_TYPE)
test.head(5)

# %%
print("Size test before filter", test.shape)
# Remove items with less than 100% of user coverage

if args.ignore_coverage_filter:
    total_users = test[DEFAULT_USER_COL].nunique()

    percentages = test.groupby(DEFAULT_ITEM_COL)[DEFAULT_USER_COL].nunique().reset_index()
    percentages.columns = [DEFAULT_ITEM_COL, "unique_users"]
    percentages["percent"] = (percentages["unique_users"] / total_users) * 100

    video_ids_to_keep = percentages.query("percent == 100")[DEFAULT_ITEM_COL].values

    test = test.query(DEFAULT_ITEM_COL + " in @video_ids_to_keep").copy().reset_index(drop=True)
print("Size test after filter", test.shape)

# %%
user_features, item_features = kuai.get_spark_features(
    spark, kuai.get_spark_df(spark, train_path), train_path, utils_path
)
user_features = user_features.toPandas()
user_features[list(kuai.USER_FEATURES_KUAI_PANDAS_TYPE.keys())] = (
    user_features[list(kuai.USER_FEATURES_KUAI_PANDAS_TYPE.keys())]
    .fillna(0)
    .astype(kuai.USER_FEATURES_KUAI_PANDAS_TYPE)
)

item_features = item_features.toPandas()
item_features[list(kuai.ITEM_FEATURES_KUAI_PANDAS_TYPE.keys())] = (
    item_features[list(kuai.ITEM_FEATURES_KUAI_PANDAS_TYPE.keys())]
    .fillna(0)
    .astype(kuai.ITEM_FEATURES_KUAI_PANDAS_TYPE)
)
item_features = item_features[list(kuai.ITEM_FEATURES_KUAI_PANDAS_TYPE.keys())]

# %%
spark.stop()
ray.init()

# %%
header = {
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
    "col_prediction": DEFAULT_PREDICTION_COL,
}
sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 0,
    "time_now": None,
    "timedecay_formula": False,
}

# (<model_name>, <model_class>, <params>)
MODELS = [
    ("sar-jaccard", SAR, {**sar_params, **header}),
    ("sar-cosine", SAR, {**sar_params, **{"similarity_type": "cosine"}, **header}),
    ("clickpop", ClickPOP, header),
    ("random", RandomModel, header),
    ("als", ImplicitModel, header),
    ("bpr", ImplicitModel, header),
    ("lightfm", LightFMModel, header),  # no fatures
    ("lightfmf", LightFMFModel, header),  # with features
]

# %%
def objective(config, train, alg_class, fixed_params, user_features, item_features):
    kf = KFold(n_splits=5)
    unique_users = train[DEFAULT_USER_COL].unique()
    metrics = []

    for train_index, val_index in kf.split(unique_users):
        train_fold_users = unique_users[train_index]
        val_fold_users = unique_users[val_index]
        train_fold = (
            pd.concat([
                train[train[DEFAULT_USER_COL].isin(train_fold_users)],
                # Return only historic of clicks
                train[train[DEFAULT_USER_COL].isin(val_fold_users)].query(f"~{DEFAULT_ISIMPR_COL}")
            ])
            .groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])
            .agg({DEFAULT_RATING_COL: "sum"})
            .reset_index()
        )
        val_fold = train[train[DEFAULT_USER_COL].isin(val_fold_users)].query(DEFAULT_ISIMPR_COL)
        all_params = {**fixed_params, **config}
        model = alg_class(**all_params)
        if hasattr(model, "fitf"):
            model.fitf(train_fold, user_features, item_features)
        else:
            model.fit(train_fold)
        result = run_eval_as_ranking(
            model, train_fold, val_fold, metrics=[TUNING_METRIC], sampler_type="cncrandom", sample_size=100
        )
        metrics.append(result[TUNING_METRIC])

    # Average the metric across folds
    avg_metric = np.mean(metrics)

    return {TUNING_METRIC: avg_metric}

# %%
for i, (alg_name, alg_class, alg_params) in enumerate(MODELS):
    train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
    model_train_path = os.path.join(model_path, train_key)
    os.makedirs(model_train_path, exist_ok=True)

    tuning_name = f"{alg_name}_tuning"
    tuning_path = os.path.join(model_train_path, tuning_name)
    if hasattr(alg_class, "get_raytune_space"):
        objective_fn = ray.tune.with_parameters(
            ray.tune.with_resources(objective, {"cpu": 4, "gpu": 0.125}),
            train=train, alg_class=alg_class, fixed_params=alg_params,
            user_features=user_features, item_features=item_features,
        )
        tuner = None
        abs_tuning_path = os.path.abspath(tuning_path)
        best_params_path = os.path.join(tuning_path, "final_best_params.json")
        best_params = {}
        if not os.path.exists(best_params_path):
            if ray.tune.Tuner.can_restore(abs_tuning_path):
                tuner = ray.tune.Tuner.restore(abs_tuning_path, objective_fn)
            else:
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

            if tuner is None:
                raise Exception("Tuning can't can be initiated...")

            tuning_results = tuner.fit()
            best_params = tuning_results.get_best_result().config
            with open(best_params_path, "w") as writer:
                json.dump(best_params, writer)
        else:
            with open(best_params_path, "r") as reader:
                best_params = json.load(reader)
        MODELS[i] = (alg_name, alg_class, {**alg_params, **best_params})

# %%
train_fold = (
    # Adds historic of clicks from users only in test set (validation (impressions) still preserved)
    pd.concat([train, test.query(f"~{DEFAULT_ISIMPR_COL}")])
    .groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    .agg({DEFAULT_RATING_COL: "sum"})
    .reset_index()
)
for alg_name, alg_class, alg_params in MODELS:
    train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
    model_train_path = os.path.join(model_path, train_key)
    os.makedirs(model_train_path, exist_ok=True)

    model = alg_class(**alg_params)
    if not model.load_model(model_train_path):
        if hasattr(model, "fitf"):
            model.fitf(train_fold, user_features, item_features)
        else:
            model.fit(train_fold)
        model.save_model(model_train_path)

# %%
def run_predict(
    sampling_exp_type, sample_size, model_train_path, alg_name, alg_class, alg_params,
    train, test, mar, result_key, results_file,
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

    print("Start", alg_name, sampling_exp_type, sample_size, result_key)

    test_with_preds = run_eval_as_ranking(
        model_to_eval, train, test.copy(), METRICS, sampling_exp_type, sample_size,
        mar=mar, byline=True, returndf=True,
    )
    test_with_preds["result_key"] = result_key

    # Save results to file
    test_with_preds.to_parquet(results_file)
    del test_with_preds
    auto_garbage_collect()

    print("End", alg_name, sampling_exp_type, sample_size)

remote_run_predict = ray.remote(run_predict)

def auto_garbage_collect(pct=80.0):
    """
    Call the garbage collection if memory used is greater than 80% of total available memory.
    This is called to deal with an issue in Ray not freeing up used memory.

    pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

# %%
rnd = np.random.RandomState(42)
np.random.seed(42)

# %%
item_bias_df = pd.DataFrame(
    pd.concat([train.query(DEFAULT_ISIMPR_COL), test.query(DEFAULT_ISIMPR_COL)])
    .groupby([DEFAULT_ITEM_COL, DEFAULT_RATING_COL]).size()
    .unstack(DEFAULT_RATING_COL, fill_value=0)
    .to_records()
).rename({"0": "negative", "1": "positive"}, axis=1)

# Random Bias
item_bias_df["rnd_prob"] = rnd.uniform(0, 1, item_bias_df.shape[0])
item_bias_df["rnd_prob"] /= item_bias_df["rnd_prob"].sum()

# Popularity Bias
item_bias_df["total"] = item_bias_df["positive"] + item_bias_df["negative"]
item_bias_df["rank"] = item_bias_df["total"].rank(ascending=False, method="min").astype(int)
item_bias_df["pop_prob"] = item_bias_df["rank"].apply(lambda x, Math=Math: Math.zipf(x+1))
item_bias_df["pop_prob"] /= item_bias_df["pop_prob"].sum()

# Positivity Bias
item_bias_df["pos_prob"] = item_bias_df["positive"] / item_bias_df["positive"].sum()

def rnd_choice(r, rnd, size):
    choices = np.zeros(r.shape[0], dtype=bool)
    ones_indexes = rnd.choice(r.index, size=size, replace=False, p=r.values)
    choices[r.index.get_indexer(ones_indexes)] = True
    return choices

# %%
RANDOM_VARIATIONS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
BIAS_PERCENTAGE = np.arange(0.0, 1.0, 0.1).tolist() + [0.85, 0.95]
BIAS_MODELS = ["random", "popularity", "positivity", "marpop", "marpos"]

# Sample MAR items
unique_item_ids = test[DEFAULT_ITEM_COL].unique().tolist()
# Reserver 10% of items to MAR
mar_unique_item_ids = rnd.choice(unique_item_ids, int(len(unique_item_ids) * 0.1), replace=False)
mar = test.query(DEFAULT_ITEM_COL + " in @mar_unique_item_ids").copy()
mar_rayid = None

futures = []
train_rayid = ray.put(train)
for bias_model in BIAS_MODELS:
    for bias_p in BIAS_PERCENTAGE:
        biased_test = test.copy()

        if bias_model in ["marpop", "marpos"]:
            # Remove MAR items
            biased_test = biased_test.query(DEFAULT_ITEM_COL + " not in @mar_unique_item_ids")
            if mar_rayid is None:
                mar_rayid = ray.put(mar)
        else:
            if mar_rayid is not None:
                rayfree([mar_rayid])
            mar_rayid = None

        biased_test_prob = biased_test.reset_index().merge(item_bias_df, on=DEFAULT_ITEM_COL, how="inner").set_index("index")
        biased_test_prob.index.name = None
        biased_test_prob["norm_prob"] = 0.0
        if bias_model == "random":
            biased_test_prob["norm_prob"] = (
                biased_test_prob.groupby(DEFAULT_USER_COL)["rnd_prob"]
                .transform(lambda r: r / r.sum())
            )
            # bias_indexes = rnd.choice(biased_test.index, int(bias_p * biased_test.shape[0]))
        elif bias_model in ["popularity", "marpop"]:
            biased_test_prob["norm_prob"] = (
                biased_test_prob.groupby(DEFAULT_USER_COL)["pop_prob"]
                .transform(lambda r: r / r.sum())
            )
        elif bias_model in ["positivity", "marpos"]:
            biased_test_prob["norm_prob"] = (
                biased_test_prob.groupby(DEFAULT_USER_COL)["pos_prob"]
                .transform(lambda r: r / r.sum())
            )

        # Keep at least 1 positive rating for each user
        positive_test = biased_test_prob[biased_test_prob[DEFAULT_RATING_COL] == 1]
        one_rating_per_user = (
            positive_test
            .groupby(DEFAULT_USER_COL)
            .sample(n=1, random_state=rnd, weights=positive_test["norm_prob"])
        )

        # Remove rows in a probabilistic way
        biased_test_prob["choice"] = (
            biased_test_prob
            .groupby(DEFAULT_USER_COL)["norm_prob"]
            .transform(lambda r, rnd=rnd: rnd_choice(r, rnd, size=int((1.0 - bias_p) * len(r))))
        )
        bias_indexes = biased_test_prob.query("choice").index.tolist()

        biased_test_final = biased_test.loc[biased_test.index.isin(bias_indexes)]

        # Check users without positive ratings
        users_with_no_rating = (
            biased_test_final
            .groupby(DEFAULT_USER_COL)
            .filter(lambda x: (x["rating"] == 1).sum() == 0)[DEFAULT_USER_COL].unique().tolist()
        )

        # Keep at least 1 positive rating for each user
        if len(users_with_no_rating) > 0:
            keep_ids = one_rating_per_user.query(f"{DEFAULT_USER_COL} in @users_with_no_rating").index.tolist()
            bias_indexes_atleast = bias_indexes + keep_ids
            biased_test_final = biased_test.loc[biased_test.index.isin(bias_indexes_atleast)]

        test_rayid = ray.put(biased_test_final)
        for alg_name, alg_class, alg_params in MODELS:
            train_key = f"{alg_name}_{SAMPLING_TRAIN_TYPE}"
            model_train_path = os.path.join(model_path, train_key)

            for sampling_exp_type in Samplers:
                variations = [None]
                if sampling_exp_type.name.lower().endswith("random"):
                    variations = RANDOM_VARIATIONS

                if sampling_exp_type in [Samplers.POPULARITY, Samplers.POSITIVITY]:
                    variations = RANDOM_VARIATIONS[1:]

                if sampling_exp_type in [Samplers.SKEW, Samplers.WTD, Samplers.WTDH]:
                    variations = RANDOM_VARIATIONS[1:]

                for sample_size in variations:
                    exp_key = f"{alg_name}_{sampling_exp_type.name}"
                    result_key = f"train:{train_key}-test_{bias_model}{int(bias_p * 100)}:{exp_key}"
                    if sample_size is not None:
                        result_key += f"@{sample_size}"
                    results_file = os.path.join(results_path, result_key + ".parquet")

                    if os.path.exists(results_file):
                        continue

                    # Enable sequentially running
                    # run_predict(
                    #     sampling_exp_type, sample_size, model_train_path,
                    #     alg_name, alg_class, alg_params,
                    #     train, biased_test_final, mar if mar_rayid is not None else None,
                    #     result_key, results_file
                    # )

                    # Submit tasks to ray
                    num_cpus = 4
                    future = remote_run_predict.options(num_cpus=num_cpus, num_gpus=0.125).remote(
                        sampling_exp_type, sample_size, model_train_path,
                        alg_name, alg_class, alg_params,
                        train_rayid, test_rayid, mar_rayid,
                        result_key, results_file
                    )
                    futures.append(future)
        rayfree([test_rayid])

# Wait for all the tasks to complete and get results
print("Waiting futures...")
results = ray.get(futures)

# %%
# Merge results to single file
print("Merging files")
results_to_merge = []
final_results_file = exec_dt + ".parquet"
final_results_path = os.path.join(results_path, exec_dt + ".parquet")

group_files = {}
for file_to_merge in os.listdir(results_path):
    try:
        group_key = file_to_merge.split("_")[2].split(":")[0]  # bias model
        group_key = ''.join([c for c in group_key if not c.isdigit()])
        if group_key not in group_files:
            group_files[group_key] = []
        group_files[group_key].append(file_to_merge)
    except IndexError:
        continue

# Save group files
for group_key, files_to_merge in group_files.items():
    results_to_merge = []
    for file_to_merge in files_to_merge:
        results_to_merge.append(pd.read_parquet(os.path.join(results_path, file_to_merge)))

    pd.concat(results_to_merge).to_parquet(final_results_path.replace(".parquet", f"_{group_key}.parquet"))

# Save single file
results_to_merge = []
for file_to_merge in os.listdir(results_path):
    if file_to_merge.startswith(exec_dt + "_"):
        results_to_merge.append(pd.read_parquet(os.path.join(results_path, file_to_merge)))
pd.concat(results_to_merge).to_parquet(final_results_path)

ray.shutdown()
