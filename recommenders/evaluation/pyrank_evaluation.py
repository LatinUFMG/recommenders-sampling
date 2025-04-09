from enum import Enum
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
import torch
from tqdm import tqdm

from . import wtd_utils as wtd


class Samplers(str, Enum):
    EXPOSED = "exposed"              # Exposed: Real negative candidates
    # FULL = "full"                  # Full: All clicked news of the day as negative candidates
    CNCFULL = "cncfull"              # Random: Sample at random a fixed N of negative candidates from all clicked news of the day
    # RAIZFULL = "raizfull"          # RandExpo: Sample at random the same number of negatives equal to real exposed from all clicked news of the day
    # RANDOM = "random"              # CNC Full: All clicked and non-clicked news of the day as negative candidates
    CNCRANDOM = "cncrandom"          # CNC Random: Sample at random a fixed N of negative candidates from all clicked and non clicked news of the day
    # RAIZRANDOM = "raizrandom"      # CNC RandExpo: Sample at random the same number of negatives equal to real exposed from all clicked and non-clicked news of the day
    # RANDEXPO = "randexpo"          # Raiz Full: All available news as negative candidates
    CNCRANDEXPO = "cncrandexpo"      # Raiz Random: Sample at random a fixed N of negative candidates from all available news
    # RAIZRANDEXPO = "raizrandexpo"  # Raiz RandExpo: Sample at random the same number of negatives equal to the real exposed from all available news
    POPULARITY = "popularity"        # Popularity: Sample from the most presented items.
    POSITIVITY = "positivity"        # Positivity: Sample from the most liked items.
    SKEW = "skew"                    # https://www.its.caltech.edu/~fehardt/UAI2016WS/papers/Liang.pdf - Close to popularity
    WTD = "wtd"                      # https://www.cs.ucc.ie/~dgb/papers/Carraro-Bridge-2019.pdf
    WTDH = "wtdh"                    # https://www.cs.ucc.ie/~dgb/papers/Carraro-Bridge-2019.pdf


class Math:
    @staticmethod
    def zipf(x, s=0.5, alpha=1):
        nominator = alpha * x ** -s
        denominator = 1
        return nominator / denominator

    @staticmethod
    def pareto(x, xm=1, alpha=1):
        return (alpha * xm ** alpha) / (x ** (alpha + 1))


def get_candidates(
    imprs, labels, full_candidates, user_weight, item_weights, sampler_type, sample_size, seed=42
):
    """Return candidates to be ranked.

    Args:
        imprs (list): Impressed indexes. If impressions isn't available, pass only positive items.
        labels (list): Label for each index in imprs.
        full_candidates (list): Candidates to be sampled.
        user_weight (dict): User weight for WTD implementation.
        item_weights (dict): Items weight for WTD implementation.
        sampler_type (Samplers): Type of sampler to be used.
        sample_size (int): Size (N) of items to be sample.
        seed (int): Random state seed.

    Returns:
        numpy.ndarray: mrr scores.
    """
    imprs = np.array(imprs)
    labels = np.array(labels, dtype="int32")
    real_true = imprs[np.where(labels)[0]]

    rnd = np.random.RandomState(seed)

    if Samplers.EXPOSED in sampler_type:
        return imprs, labels
    elif "full" in sampler_type:
        sample_imprs = np.fromiter(full_candidates.index, dtype=imprs.dtype)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif "random" in sampler_type:
        fc_ids = np.fromiter(set(full_candidates.index) - set(real_true), dtype=imprs.dtype)
        sample_imprs = rnd.choice(fc_ids, size=min(sample_size, len(fc_ids)), replace=False)
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif "randexpo" in sampler_type:
        fc_ids = np.fromiter(set(full_candidates.index) - set(real_true), dtype=imprs.dtype)
        sample_imprs = rnd.choice(
            fc_ids, size=min(len(imprs) - len(real_true), len(fc_ids)), replace=False
        )
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif Samplers.POPULARITY in sampler_type:
        fc_df = full_candidates.copy()
        fc_df["total"] = fc_df["positive"] + fc_df["negative"]
        fc_df["rank"] = fc_df["total"].rank(ascending=False, method="min").astype(int)
        fc_df["prob"] = fc_df["rank"].apply(lambda x, Math=Math: Math.zipf(x+1))

        fc_df.loc[real_true]["prob"] = 0.0  # Remove true from candidates
        fc_df["prob"] /= fc_df["prob"].sum()  # Normalize weights

        sample_imprs = rnd.choice(
            fc_df.index, size=min(sample_size, len(fc_df)), replace=False, p=fc_df["prob"].values
        ).astype(imprs.dtype)
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif Samplers.POSITIVITY in sampler_type:
        fc_df = full_candidates.copy()
        fc_df["prob"] = fc_df["positive"]

        fc_df.loc[real_true]["prob"] = 0.0  # Remove true from candidates
        fc_df["prob"] /= fc_df["prob"].sum()  # Normalize weights

        sample_imprs = rnd.choice(
            fc_df.index, size=min(sample_size, len(fc_df)), replace=False, p=fc_df["prob"].values
        ).astype(imprs.dtype)
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif Samplers.SKEW in sampler_type:
        fc_df = full_candidates.copy()
        fc_df["prob"] = fc_df["positive"] + fc_df["negative"]
        fc_df.loc[real_true]["prob"] = 0.0  # Remove true from candidates
        fc_df["prob"] /= fc_df["prob"].sum()  # Normalize weights

        sample_imprs = rnd.choice(
            fc_df.index, size=min(sample_size, len(fc_df)), replace=False, p=fc_df["prob"].values
        ).astype(imprs.dtype)
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    elif Samplers.WTD in sampler_type or Samplers.WTDH in sampler_type:
        # Calcule P(S|u,i,w) for every pair user/candidate
        p_s = np.zeros(len(full_candidates))
        for i, candidate in enumerate(full_candidates.index):
            if candidate in real_true:
                p_s[i] = 0.0
            else:
                p_s[i] = user_weight * item_weights[candidate]

        # Normalize P(S|u,i,w)
        p_s /= np.sum(p_s)

        sample_imprs = rnd.choice(
            full_candidates.index, size=min(sample_size, len(full_candidates)), replace=False, p=p_s
        )
        sample_imprs = np.concatenate([sample_imprs, real_true])
        rnd.shuffle(sample_imprs)

        # Amostragem com a distribuição ponderada
        sample_gt_labels_idxs = [np.where(sample_imprs == gt)[0][0] for gt in real_true]
        sample_labels = np.zeros(len(sample_imprs), dtype="int32")
        sample_labels[sample_gt_labels_idxs] = 1

        return sample_imprs, sample_labels
    else:
        raise ValueError(f"Got unexpected sampler: {sampler_type}.")

    # TODO: Bias-Adjusted model
    # - Choose an unbiased recommender algorithm and train the model using train/test logs
    # - Save model using pkl for training once
    # - Load model
    # - Predict self.test_sample_size candidates


def get_full_candidate(test, sampler_type, col_item, col_rating, col_isimpr):
    """Get the full candidate items for evaluation.
    Args:
        test (pd.DataFrame): A pandas dataframe with validation/test set.
        sampler_type (str): Type of sampler to be used.
        col_isimpr (str): Column indicates if the item was actually impressed to the user.
        col_item (str): The item column.
        col_rating (str): The rating column.
    Returns:
        pd.DataFrame: DataFrame with candidates ids, number of positive and negative feedbacks.
    """
    return pd.DataFrame(
        test
        .astype({col_rating: "str"})
        .groupby([col_isimpr, col_item, col_rating]).size()
        .unstack(col_rating, fill_value=0)
        .to_records()
    ).rename({"0": "negative", "1": "positive"}, axis=1)


def _cast_to_native_type(x):
    if type(x).__module__ == np.__name__:
        return x.item()
    else:
        return x


def cast_to_native_type(x):
    cast_values = []
    if hasattr(x, "__iter__") and not isinstance(x, str):
        for tocast in x:
            cast_values.append(_cast_to_native_type(tocast))
        return cast_values
    else:
        return _cast_to_native_type(x)


def run_eval_as_ranking(model, train, test, metrics, sampler_type, sample_size=None, mar=None, byline=False, returndf=False):
        """Evaluate the model as a ranker.
        Args:
            model (any): Model to be evaluated.
            train (pd.DataFrame): A pandas dataframe with training set.
            test (pd.DataFrame): A pandas dataframe with validation/test set.
            metrics (list): List of metrics (implemented by utils.metrics).
            sampler_type (str): Type of sampler to be used.
            sample_size (int): Size (N) of items to be sampled.
            mar (pd.DataFrame): A pandas dataframe with the missing at random data.
            byline (bool): If the return should be by line or aggregated. Defaults is False.
            returndf (bool): If the return is a dataframe instead metrics in dict. byline True is required. Defaults is False.
        Returns:
            dict: A dictionary that contains evaluation metrics.
            pd.DataFrame: DataFrame with labels, pedictions, metrics...
        """
        # Transform the output of "get_full_candidate" and return it as a set
        candidates_to_sample = get_full_candidate(
            test, sampler_type, model.col_item, model.col_rating, model.col_isimpr
        )

        pscore_dict = {}
        if train is not None:
            pscore_indexes, item_freq = np.unique(
                test.query(f"{model.col_rating} == 1")[model.col_item], return_counts=True
            )
            pscore = (item_freq / item_freq.max()) ** 0.5
            pscore_dict = dict(zip(pscore_indexes, pscore))

        user_weights = {}
        item_weights = {}
        use_ideal = False
        if sampler_type in [Samplers.WTD, Samplers.WTDH]:
            print("Calculating weights for WTD sampler")
            mar_df = mar.copy() if mar is not None else test.copy()
            if sampler_type == Samplers.WTDH:
                use_ideal = True
                mar_df = test.copy()
            user_weights, item_weights = wtd.calculate_weights(
                test, mar_df, use_ideal,
                test[model.col_user].unique().tolist(), test[model.col_item].unique().tolist()
            )

        candidates_to_sample = candidates_to_sample.query(f"{model.col_isimpr}")
        candidates_to_sample = candidates_to_sample.drop([model.col_isimpr], axis=1).set_index(model.col_item)

        tests_group = (
            test
            .query(f"{model.col_isimpr}")
            .groupby([model.col_user, model.col_imprgroup])
            .agg({model.col_item: lambda x: list(x), model.col_rating: lambda x: list(x)})
            .reset_index()
        )

        list_returndf = []

        # Get unique users and iterate in chunks of 100
        unique_users = tests_group[model.col_user].unique()

        chunk_size = 2000
        for i in tqdm(range(0, len(unique_users), chunk_size)):
            user_chunk = unique_users[i:i+chunk_size]

            # Filter the tests_group dataframe to get only rows with users in the current chunk
            chunk_df = tests_group[tests_group[model.col_user].isin(user_chunk)].copy()

            # Make a copy of columns col_item and col_rating
            chunk_df[model.col_item + "_bkp"] = chunk_df[model.col_item]
            chunk_df[model.col_rating + "_bkp"] = chunk_df[model.col_rating]

            # Apply the get_candidates function to each row
            chunk_df[[model.col_item, model.col_rating]] = chunk_df.apply(
                lambda r: get_candidates(
                    r[model.col_item], r[model.col_rating], candidates_to_sample,
                    user_weights.get(r[model.col_user]), item_weights,  # WTD parameters
                    sampler_type, sample_size,
                    int(re.sub("[^0-9]", "", str(r[model.col_user])) + "42")
                ), axis=1, result_type="expand"
            )

            # Predict for the entire chunk at once
            predictions = model.predict(chunk_df.explode([model.col_item, model.col_rating]))

            # Map the propensity score
            predictions["pscore"] = predictions[model.col_item].map(pscore_dict)

            # Append the results to group_labels and group_preds
            test_labels_preds = (
                predictions
                .groupby([model.col_imprgroup, model.col_user])
                .agg({
                    model.col_item: lambda x: list(x),
                    model.col_rating: lambda x: list(x),
                    model.col_prediction: lambda x: list(x),
                    "pscore": lambda x: list(x),
                    # model.col_item + "_bkp": "first",
                    # model.col_rating + "_bkp": "first",
                })
                .reset_index()
            )
            group_labels = [cast_to_native_type(x) for x in test_labels_preds.values[:, 3]]
            group_preds = [cast_to_native_type(x) for x in test_labels_preds.values[:, 4]]
            group_pscore = [cast_to_native_type(x) for x in test_labels_preds.values[:, 5]]

            evaluation_metrics = [
                cal_metric(group_labels[i:i+1], group_preds[i:i+1], metrics, group_pscore[i:i+1])
                for i in range(len(group_labels))
            ]
            test_labels_preds["evaluation"] = evaluation_metrics
            test_labels_preds["size_candidates"] = test_labels_preds[model.col_item].str.len()
            test_labels_preds["sum_rating"] = test_labels_preds[model.col_rating].apply(sum)
            test_labels_preds = (
                test_labels_preds
                .drop([model.col_item, model.col_prediction, model.col_rating, "pscore"], axis=1)
                # .rename({
                #     model.col_item + "_bkp": model.col_item,
                #     model.col_rating + "_bkp": model.col_rating
                # }, axis=1)
            )
            list_returndf.append(test_labels_preds)

        returndf_final = pd.concat(list_returndf)
        if byline:
            if returndf:
                return returndf_final
            else:
                return returndf_final["evaluation"].tolist()
        else:
            dict_df = pd.DataFrame(returndf_final["evaluation"].tolist()).fillna(0)
            return dict_df.mean().to_dict()


def mrr_score(y_true, y_score, k=None):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.
        k(int): The number of top predictions to consider for MRR calculation.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)

    if k is not None:
        # Truncate to the top k predictions
        y_true = y_true[:k]

    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) != 0 else 0


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def unbiased_dcg_score(
    y_true: list, y_score: list, k: int, pscore: list = None
) -> float:
    """Calculate a DCG score for a given user."""
    eps = 10e-3  # pscore clipping
    y_true_sorted_by_score = np.take(y_true, np.argsort(y_score)[::-1])

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            np.take(np.nan_to_num(pscore), np.argsort(y_score)[::-1]), eps
        )
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    dcg_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        dcg_score += y_true_sorted_by_score[0] / pscore_sorted_by_score[0]
        for i in np.arange(1, min(k, len(y_true_sorted_by_score))):
            dcg_score += (
                y_true_sorted_by_score[i] / (pscore_sorted_by_score[i] * np.log2(i + 1))
            )

    if pscore is None:
        return dcg_score
    else:
        return dcg_score / np.sum(1.0 / pscore_sorted_by_score[y_true_sorted_by_score == 1])


def unbiased_precision_at_k(
    y_true: list, y_score: list, k: int, pscore: list = None
) -> float:
    """Calculate a average precision for a given user."""
    eps = 10e-3  # pscore clipping
    y_true_sorted_by_score = np.take(y_true, np.argsort(y_score)[::-1])

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            np.take(np.nan_to_num(pscore), np.argsort(y_score)[::-1]), eps
        )
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    average_precision_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        for i in np.arange(1, min(k, len(y_true_sorted_by_score))):
            if y_true_sorted_by_score[i] == 1:
                average_precision_score += \
                    np.sum(y_true_sorted_by_score[:i + 1] /
                           pscore_sorted_by_score[:i + 1]) / (i + 1)

    if pscore is None:
        return average_precision_score
    else:
        return average_precision_score / np.sum(1. / pscore_sorted_by_score[y_true_sorted_by_score == 1])


def precision_score_at_k(y_true, y_score, k=10):
    y_true = np.asarray(y_true)
    # Sort predictions and labels based on the scores in descending order
    sorted_indices = np.argsort(y_score)[::-1][:k]
    top_k_labels = y_true[sorted_indices]

    return np.sum(top_k_labels > 0) / top_k_labels.shape[0]


def unbiased_recall_at_k(
    y_true: list, y_score: list, k: int, pscore: list = None
) -> float:
    """Calculate a recall score for a given user."""
    eps = 10e-3  # pscore clipping
    y_true_sorted_by_score = np.take(y_true, np.argsort(y_score)[::-1])

    if pscore is not None:
        pscore_sorted_by_score = np.maximum(
            np.take(np.nan_to_num(pscore), np.argsort(y_score)[::-1]), eps
        )
    else:
        pscore_sorted_by_score = np.ones_like(y_true_sorted_by_score)

    recall_score = 0.0
    if not np.sum(y_true_sorted_by_score) == 0:
        recall_score = np.sum(y_true_sorted_by_score[:k] / pscore_sorted_by_score[:k])

    if pscore is None:
        return recall_score / np.sum(y_true)
    else:
        return recall_score / np.sum(1.0 / pscore_sorted_by_score[y_true_sorted_by_score == 1])


def recall_score_at_k(y_true, y_score, k=10):
    y_true = np.asarray(y_true)
    # Sort predictions and labels based on the scores in descending order
    sorted_indices = np.argsort(y_score)[::-1][:k]
    top_k_labels = y_true[sorted_indices]

    return np.sum(top_k_labels > 0) / np.sum(y_true > 0)


def cal_metric(labels, preds, metrics, pscore=None):
    """Calculate metrics.

    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.
        metrics (list): List of metric names to be generated.
        pscore (array-like): Item propensitity for ubiased metric calculation.

    Return:
        dict: Metrics.

    Examples:
        >>> cal_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])
        {'ndcg@2': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = None
            if len(np.unique(labels)) > 1.0:
                auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric.startswith("mean_mrr"):
            ks = metric.split("@")
            if len(ks) > 1:
                mrr_list = [int(token) for token in ks[1].split(";")]
                for k in mrr_list:
                    mrr_temp = np.mean(
                        [
                            mrr_score(each_labels, each_preds, k)
                            for each_labels, each_preds in zip(labels, preds)
                        ]
                    )
                    res["mean_mrr@{0}".format(k)] = round(mrr_temp, 4)
            else:
                mean_mrr = np.mean(
                    [
                        mrr_score(each_labels, each_preds)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("unbiased_ndcg"):
            # Unbiased NDCG using IPS
            if pscore is None:
                raise ValueError("Propensities are required for unbiased metrics.")

            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                unbiased_ndcg_temp = np.mean(
                    [
                        unbiased_dcg_score(each_labels, each_preds, k, each_propensities)
                        for each_labels, each_preds, each_propensities in zip(labels, preds, pscore)
                    ]
                )
                res["unbiased_ndcg@{0}".format(k)] = round(unbiased_ndcg_temp, 4)

        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                    if len(np.unique(each_labels)) > 1
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        elif metric.startswith("precision"):
            precision_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                precision_list = [int(token) for token in ks[1].split(";")]
            for k in precision_list:
                precision_temp = np.mean(
                    [
                        precision_score_at_k(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["precision@{0}".format(k)] = round(precision_temp, 4)
        elif metric.startswith("unbiased_precision"):
            # Unbiased precision using IPS
            if pscore is None:
                raise ValueError("Propensities are required for unbiased metrics.")

            prec_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                prec_list = [int(token) for token in ks[1].split(";")]
            for k in prec_list:
                unbiased_prec_temp = np.mean(
                    [
                        unbiased_precision_at_k(each_labels, each_preds, k, each_propensities)
                        for each_labels, each_preds, each_propensities in zip(labels, preds, pscore)
                    ]
                )
                res["unbiased_precision@{0}".format(k)] = round(unbiased_prec_temp, 4)
        elif metric.startswith("recall"):
            recall_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                recall_list = [int(token) for token in ks[1].split(";")]
            for k in recall_list:
                recall_temp = np.mean(
                    [
                        recall_score_at_k(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["recall@{0}".format(k)] = round(recall_temp, 4)
        elif metric.startswith("unbiased_recall"):
            # Unbiased recall using IPS
            if pscore is None:
                raise ValueError("Propensities are required for unbiased metrics.")

            rec_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                rec_list = [int(token) for token in ks[1].split(";")]
            for k in rec_list:
                unbiased_rec_temp = np.mean(
                    [
                        unbiased_recall_at_k(each_labels, each_preds, k, each_propensities)
                        for each_labels, each_preds, each_propensities in zip(labels, preds, pscore)
                    ]
                )
                res["unbiased_recall@{0}".format(k)] = round(unbiased_rec_temp, 4)
        else:
            raise ValueError("Metric {0} not defined".format(metric))
    return res
