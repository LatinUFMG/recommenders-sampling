import json
import os
import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from ray import tune
from scipy import sparse
import threadpoolctl

from recommenders.utils import constants


threadpoolctl.threadpool_limits(1, "blas")


class ImplicitModel:
    def __init__(
            self,
            col_user=constants.DEFAULT_USER_COL,
            col_item=constants.DEFAULT_ITEM_COL,
            col_rating=constants.DEFAULT_RATING_COL,
            col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
            col_prediction=constants.DEFAULT_PREDICTION_COL,
            col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
            col_isimpr=constants.DEFAULT_ISIMPR_COL,
            model_type='als', factors=10, iterations=10, regularization=0.1,
            alpha=1,  # ALS param
            learning_rate=0.01,  # BPR param
            use_gpu=implicit.gpu.HAS_CUDA,
            agg_func='sum',
            *args, **kwargs
        ):
        """
        Initialize the model.

        Args:
            col_user (str): Name of the user column in DataFrames.
            col_item (str): Name of the item column in DataFrames.
            col_rating (str): Name of the rating column in DataFrames.
            col_timestamp (str): Name of the timestamp column in DataFrames.
            col_prediction (str): Name of the prediction column in DataFrames.
            col_imprgroup (str): Name of the impression group column in DataFrames.
            col_isimpr (str): Name of the is impression column in DataFrames.
            model_type (str): The type of implicit model ('als' or 'bpr').
            factors (int): Number of factors for ALS or BPR.
            iterations (int): Number of iterations for training.
            regularization (float): Regularization parameter for ALS or BPR.
            alpha (float): Confidence parameter for ALS.
            learning_rate (float): Learning rate parameter for BPR.
            use_gpu (bool): Enable usage of GPU.
            agg_func(str): Aggregation function for duplicated entries in train dataset.
        """
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        self.col_imprgroup = col_imprgroup
        self.col_isimpr = col_isimpr

        self.model_type = model_type
        if self.model_type == 'als':
            self.model = AlternatingLeastSquares(
                factors=factors, regularization=regularization, alpha=alpha, iterations=iterations,
                use_gpu=use_gpu
            )
        elif self.model_type == 'bpr':
            self.model = BayesianPersonalizedRanking(
                factors=factors, regularization=regularization, learning_rate=learning_rate,
                iterations=iterations, use_gpu=use_gpu
            )
        else:
            raise ValueError("Invalid model type. Choose 'als' or 'bpr'.")

        self.agg_func = agg_func
        self.user_item_matrix = None

        # column for mapping user / item ids to internal indices
        self.col_item_id = "_indexed_items"
        self.col_user_id = "_indexed_users"

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above maps - map array index to actual string ID
        self.index2item = None
        self.index2user = None

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage.

        Args:
            df (pandas.DataFrame): dataframe with user and item ids
        """

        # generate a map of continuous index values to items
        self.index2item = dict(enumerate(df[self.col_item].unique().astype("str").tolist()))
        self.index2user = dict(enumerate(df[self.col_user].unique().astype("str").tolist()))

        # invert the mappings from above
        self.item2index = {v: k for k, v in self.index2item.items()}
        self.user2index = {v: k for k, v in self.index2user.items()}

        # set values for the total count of users and items
        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

    @staticmethod
    def get_hyperopt_space(model_type: str = "als"):
        default_space = {
            "factors": scope.int(hp.quniform("factors", 10, 200, 1)),
            "iterations": scope.int(hp.quniform("iterations", 10, 100, 1)),
            "regularization": hp.loguniform("regularization", -4, 0),
        }
        if model_type == "als":
            default_space["model_type"] = hp.choice("model_type", ["als"])
            default_space["alpha"] = hp.loguniform("alpha", -4, 2)
            return default_space
        elif model_type == "bpr":
            default_space["model_type"] = hp.choice("model_type", ["bpr"])
            default_space["learning_rate"] = hp.loguniform("learning_rate", -4, 0)
            return default_space
        else:
            raise NotImplementedError()

    @staticmethod
    def get_raytune_space(model_type: str = "als"):
        default_space = {
            "factors": tune.randint(10, 201),
            "iterations": tune.randint(10, 51),
            "regularization": tune.loguniform(1e-5, 10.0),
        }
        if model_type == "als":
            default_space["model_type"] = tune.choice(["als"])
            default_space["alpha"] = tune.randint(1, 101)
            return default_space
        elif model_type == "bpr":
            default_space["model_type"] = tune.choice(["bpr"])
            default_space["learning_rate"] = tune.loguniform(1e-4, 0.1)
            default_space["neg_sample"] =  tune.randint(1, 6)
            return default_space
        else:
            raise NotImplementedError()

    def to_gpu(self):
        try:
            self.model = self.model.to_gpu()
        except Exception as ex:
            print("Can't convert model to GPU", ex)

    def fit(self, df: pd.DataFrame):
        """Main fit method for ImplicitModel.

        Note:
            Please make sure that `df` has no duplicates.

        Args:
            df (pd.DataFrame): User item rating dataframe (without duplicates).
        """
        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        # copy the DataFrame to avoid modification of the input
        temp_df = df.copy()

        # add mapping of user and item ids to indices
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].apply(
            lambda user: self.user2index.get(str(user), -1)
        )
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].apply(
            lambda item: self.item2index.get(str(item), -1)
        )

        temp_df = temp_df.groupby([self.col_user_id, self.col_item_id]).agg(
            {self.col_rating: self.agg_func}
        ).reset_index()
        user_ids = temp_df[self.col_user_id].values
        item_ids = temp_df[self.col_item_id].values
        ratings = temp_df[self.col_rating].values

        self.user_item_matrix = sparse.csr_matrix((ratings, (user_ids, item_ids)))
        self.model.fit(self.user_item_matrix)

    def score(self, test, remove_seen=False):
        """Score all items for test users.

        Args:
            test (pd.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        user_ids = test[self.col_user].apply(
            lambda user: self.user2index.get(str(user), -1)
        ).values.tolist()
        scores = np.zeros((len(user_ids), self.user_item_matrix.shape[1]))

        for i, user_id in enumerate(user_ids):
            if user_id == -1:
                user_scores = [(k, 0.0) for k in self.item2index]
            else:
                user_scores = self.model.recommend(
                    user_id, self.user_item_matrix[user_id], N=self.user_item_matrix.shape[1],
                    filter_already_liked_items=remove_seen
                )
            scores[i, [item for item, _ in user_scores]] = [score for _, score in user_scores]

        return scores

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pandas.DataFrame): users to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pandas.DataFrame: top k recommendation items for each user
        """
        recommendations = []
        user_ids = test[self.col_user].apply(
            lambda user: self.user2index.get(str(user), -1)
        ).values.tolist()

        for user_id in user_ids:
            if user_id == -1:
                user_scores = [(k, 0.0) for k in list(self.item2index)[:top_k]]
            else:
                user_scores = self.model.recommend(
                    user_id, self.user_item_matrix[user_id], N=top_k,
                    filter_already_liked_items=remove_seen
                )

            if sort_top_k:
                user_scores.sort(key=lambda x: x[1], reverse=True)

            recommendations.extend([(user_id, item, score) for item, score in user_scores[:top_k]])

        return pd.DataFrame(recommendations, columns=[self.col_user, self.col_item, self.col_prediction])

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set

        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test

        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """
        # TODO: Atualizar funções anteriores

        # add mapping of user and item ids to indices
        test.loc[:, self.col_user_id] = test[self.col_user].apply(
            lambda user: self.user2index.get(str(user), -1)
        )
        test.loc[:, self.col_item_id] = test[self.col_item].apply(
            lambda item: self.item2index.get(str(item), -1)
        )

        topredict = (
            test
            .query(f"{self.col_user_id} > -1 and {self.col_item_id} > -1")
            .groupby([self.col_user_id]).agg({self.col_item_id: lambda x: set(x)})
            .reset_index()
        )
        predictions = {self.col_user_id: [], self.col_item_id: [], self.col_prediction: []}
        for _, (user_id, items) in topredict.iterrows():
            scored_items, scores = self.model.recommend(
                user_id, self.user_item_matrix[user_id], len(items), items=list(items),
                filter_already_liked_items=False
            )
            predictions[self.col_user_id].extend([user_id] * len(scored_items))
            predictions[self.col_item_id].extend(scored_items)
            predictions[self.col_prediction].extend(scores)

        res = test.merge(pd.DataFrame(predictions), on=[self.col_user_id, self.col_item_id], how="left")
        res[self.col_prediction] = res[self.col_prediction].fillna(0.0)

        res.drop([self.col_user_id, self.col_item_id], axis=1, inplace=True)

        return res

    def save_model(self, filepath):
        os.makedirs(filepath, exist_ok=True)

        # Basic attributes
        basic_attr = {
            "index2item": self.index2item,
            "n_items": self.n_items,
            "item2index": self.item2index,
            "index2user": self.index2user,
            "n_users": self.n_users,
            "user2index": self.user2index,
        }

        filename = os.path.join(filepath, f"implicit_{self.model_type}_attr.json")
        with open(filename, "w") as fout:
            json.dump({k: v for k, v in basic_attr.items() if v is not None}, fout)

        filename = os.path.join(filepath, f"implicit_{self.model_type}")
        self.model.save(filename)

        filename = os.path.join(filepath, "user_item_matrix.npz")
        sparse.save_npz(filename, self.user_item_matrix)

    def load_model(self, filepath):
        """Load the model from a file.
        Args:
            filepath (str): Path to load the model from.
        Returns:
            bool: True if model successfully loaded, otherwise False.
        """
        try:
            filename = os.path.join(filepath, f"implicit_{self.model_type}_attr.json")
            with open(filename, "r") as fin:
                basic_attr = json.load(fin)
                for k, v in basic_attr.items():
                    setattr(self, k, v)

            filename = os.path.join(filepath, f"implicit_{self.model_type}")
            self.model = self.model.load(filename)

            filename = os.path.join(filepath, "user_item_matrix.npz")
            self.user_item_matrix = sparse.load_npz(filename)
            return True
        except Exception as ex:
            print(ex)
            return False
