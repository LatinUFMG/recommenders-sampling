import json
import os
import pandas as pd

from recommenders.utils import constants

class ClickPopularityModel:
    """Popularity Model that counts the number of times
    each item was clicked on in the training data."""

    def __init__(
            self,
            col_user=constants.DEFAULT_USER_COL,
            col_item=constants.DEFAULT_ITEM_COL,
            col_rating=constants.DEFAULT_RATING_COL,
            col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
            col_prediction=constants.DEFAULT_PREDICTION_COL,
            col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
            col_isimpr=constants.DEFAULT_ISIMPR_COL,
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
        self.item_clicks = None

    def fit(self, df):
        """Fit the model by counting the number of clicks for each item.
        Args:
            df (pandas.DataFrame): User item rating dataframe.
        """
        group_df = (
            df
            .groupby(self.col_item)
            .agg({self.col_rating: "sum"})
            .reset_index()
        )
        group_df[self.col_item] = group_df[self.col_item].astype(str)
        self.item_clicks = group_df.set_index(self.col_item)[self.col_rating].to_dict()

    def score(self, test, remove_seen=False):
        """Score all items for test users.
        Args:
            test (pandas.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation
        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        if self.item_clicks is None:
            raise ValueError("Model has not been fitted yet. Call the fit method first.")

        user_ids = test[self.col_user].unique()
        scores = []
        for _ in user_ids:
            user_scores = {**self.item_clicks}
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

    def predict(self, test):
        """Output popularity scores for only the users-items pairs which are in the test set
        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test
        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """
        if self.item_clicks is None:
            raise ValueError("Model has not been fitted yet. Call the fit method first.")
        test[self.col_prediction] = test[self.col_item].astype(str).map(self.item_clicks).fillna(0)
        return test

    def save_model(self, filepath):
        """Save the model to a file.
        Args:
            filepath (str): Path to save the model.
        """
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, "item_clicks.json")
        with open(filename, "w") as fout:
            json.dump(self.item_clicks, fout)

    def load_model(self, filepath):
        """Load the model from a file.
        Args:
            filepath (str): Path to load the model from.
        Returns:
            bool: True if model successfully loaded, otherwise False.
        """
        try:
            filename = os.path.join(filepath, "item_clicks.json")
            with open(filename, "r") as fin:
                self.item_clicks = json.load(fin)
                return True
        except Exception as ex:
            print(ex)
            return False
