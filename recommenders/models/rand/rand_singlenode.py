import os
import pandas as pd
import numpy as np

from recommenders.utils import constants

class RandomModel:
    """Random Model."""

    def __init__(
            self,
            col_user=constants.DEFAULT_USER_COL,
            col_item=constants.DEFAULT_ITEM_COL,
            col_rating=constants.DEFAULT_RATING_COL,
            col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
            col_prediction=constants.DEFAULT_PREDICTION_COL,
            col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
            col_isimpr=constants.DEFAULT_ISIMPR_COL,
            seed=42,
            *args, **kwargs
        ):
        """Initialize model parameters
        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            col_itemgroup (str): column used to "group" the positive and negative candidates for ranking.
            col_isimpr (str): column used to filter real impressions from historical log.
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        self.col_imprgroup = col_imprgroup
        self.col_isimpr = col_isimpr
        self._seed = seed
        self._rnd = np.random.RandomState(self._seed)
        self.items = None

    def fit(self, df):
        """Fit the model by counting the number of clicks for each item.
        Args:
            df (pandas.DataFrame): User item rating dataframe.
        """
        self.items = df[self.col_item].unique()

    def score(self, test, remove_seen=False):
        """Score all items for test users.
        Args:
            test (pandas.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation
        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        if self.items is None:
            raise ValueError("Model has not been fitted yet. Call the fit method first.")

        user_ids = test[self.col_user].unique()
        scores = []
        for _ in user_ids:
            user_scores = {item: self._rnd.uniform(0, 1) for item in self.items}
            scores.append(user_scores)
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
        scores = self.score(test, remove_seen=remove_seen)
        recommendations = []
        for user_scores in scores:
            sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            recommendations.append(sorted_items)
        result = []
        for user, user_recommendations in zip(test[self.col_user].unique(), recommendations):
            for item, score in user_recommendations:
                result.append({self.col_user: user, self.col_item: item, self.col_prediction: score})
        return pd.DataFrame(result)

    def predict1(self, test):
        """Output popularity scores for only the users-items pairs which are in the test set
        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test
        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """
        test_with_predict = test.copy()
        test_with_predict[self.col_prediction] = self._rnd.uniform(0, 1, test_with_predict.shape[0])
        return test_with_predict

    def predict(self, test):
        """Output popularity scores for only the users-items pairs which are in the test set
        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test
        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """
        def generate_score(row):
            # Combine user and item IDs into a string and hash it to create a seed
            seed = self._seed + int(f"{row[self.col_user]}{row[self.col_item]}")
            rnd = np.random.default_rng(seed)  # Create a temporary RNG with the specific seed
            return rnd.uniform(0, 1)  # Generate a random number in [0, 1)

        test_with_predict = test.copy()
        test_with_predict[self.col_prediction] = test.apply(generate_score, axis=1)
        return test_with_predict

    def save_model(self, filepath):
        """Save the model to a file.
        Args:
            filepath (str): Path to save the model.
        """
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, "items.npy")
        np.save(filename, self.items)

    def load_model(self, filepath):
        """Load the model from a file.
        Args:
            filepath (str): Path to load the model from.
        Returns:
            bool: True if model successfully loaded, otherwise False.
        """
        try:
            filename = os.path.join(filepath, "items.npy")
            self.items = np.load(filename)
            return True
        except Exception as ex:
            print(ex)
            return False
