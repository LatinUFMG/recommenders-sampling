# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import seaborn as sns

from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm import LightFM
from lightfm.data import Dataset
from ray import tune

from recommenders.utils import constants


def model_perf_plots(df):
    """Function to plot model performance metrics.

    Args:
        df (pandas.DataFrame): Dataframe in tidy format, with ["epoch","level","value"] columns

    Returns:
        object: matplotlib axes
    """
    g = sns.FacetGrid(df, col="metric", hue="stage", col_wrap=2, sharey=False)
    g = g.map(sns.scatterplot, "epoch", "value").add_legend()


def compare_metric(df_list, metric="prec", stage="test"):
    """Function to combine and prepare list of dataframes into tidy format.

    Args:
        df_list (list): List of dataframes
        metrics (str): name of metric to be extracted, optional
        stage (str): name of model fitting stage to be extracted, optional

    Returns:
        pandas.DataFrame: Metrics
    """
    colnames = ["model" + str(x) for x in list(range(1, len(df_list) + 1))]
    models = [
        df[(df["stage"] == stage) & (df["metric"] == metric)]["value"]
        .reset_index(drop=True)
        .values
        for df in df_list
    ]

    output = pd.DataFrame(zip(*models), columns=colnames).stack().reset_index()
    output.columns = ["epoch", "data", "value"]
    return output


def track_model_metrics(
    model,
    train_interactions,
    test_interactions,
    k=10,
    no_epochs=100,
    no_threads=8,
    show_plot=True,
    **kwargs
):
    """Function to record model's performance at each epoch, formats the performance into tidy format,
    plots the performance and outputs the performance data.

    Args:
        model (LightFM instance): fitted LightFM model
        train_interactions (scipy sparse COO matrix): train interactions set
        test_interactions (scipy sparse COO matrix): test interaction set
        k (int): number of recommendations, optional
        no_epochs (int): Number of epochs to run, optional
        no_threads (int): Number of parallel threads to use, optional
        **kwargs: other keyword arguments to be passed down

    Returns:
        pandas.DataFrame, LightFM model, matplotlib axes:
        - Performance traces of the fitted model
        - Fitted model
        - Side effect of the method
    """
    # initialising temp data storage
    model_prec_train = [0] * no_epochs
    model_prec_test = [0] * no_epochs

    model_rec_train = [0] * no_epochs
    model_rec_test = [0] * no_epochs

    # fit model and store train/test metrics at each epoch
    for epoch in range(no_epochs):
        model.fit_partial(
            interactions=train_interactions, epochs=1, num_threads=no_threads, **kwargs
        )
        model_prec_train[epoch] = precision_at_k(
            model, train_interactions, k=k, **kwargs
        ).mean()
        model_prec_test[epoch] = precision_at_k(
            model, test_interactions, k=k, **kwargs
        ).mean()

        model_rec_train[epoch] = recall_at_k(
            model, train_interactions, k=k, **kwargs
        ).mean()
        model_rec_test[epoch] = recall_at_k(
            model, test_interactions, k=k, **kwargs
        ).mean()

    # collect the performance metrics into a dataframe
    fitting_metrics = pd.DataFrame(
        zip(model_prec_train, model_prec_test, model_rec_train, model_rec_test),
        columns=[
            "model_prec_train",
            "model_prec_test",
            "model_rec_train",
            "model_rec_test",
        ],
    )
    # convert into tidy format
    fitting_metrics = fitting_metrics.stack().reset_index()
    fitting_metrics.columns = ["epoch", "level", "value"]
    # exact the labels for each observation
    fitting_metrics["stage"] = fitting_metrics.level.str.split("_").str[-1]
    fitting_metrics["metric"] = fitting_metrics.level.str.split("_").str[1]
    fitting_metrics.drop(["level"], axis=1, inplace=True)
    # replace the metric keys to improve visualisation
    metric_keys = {"prec": "Precision", "rec": "Recall"}
    fitting_metrics.metric.replace(metric_keys, inplace=True)
    # plots the performance data
    if show_plot:
        model_perf_plots(fitting_metrics)
    return fitting_metrics, model


def similar_users(user_id, user_features, model, N=10):
    """Function to return top N similar users based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681

     Args:
        user_id (int): id of user to be used as reference
        user_features (scipy sparse CSR matrix): user feature matric
        model (LightFM instance): fitted LightFM model
        N (int): Number of top similar users to return

    Returns:
        pandas.DataFrame: top N most similar users with score
    """
    _, user_representations = model.get_user_representations(features=user_features)

    # Cosine similarity
    scores = user_representations.dot(user_representations[user_id, :])
    user_norms = np.linalg.norm(user_representations, axis=1)
    user_norms[user_norms == 0] = 1e-10
    scores /= user_norms

    best = np.argpartition(scores, -(N + 1))[-(N + 1) :]
    return pd.DataFrame(
        sorted(zip(best, scores[best] / user_norms[user_id]), key=lambda x: -x[1])[1:],
        columns=["userID", "score"],
    )


def similar_items(item_id, item_features, model, N=10):
    """Function to return top N similar items
    based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681

    Args:
        item_id (int): id of item to be used as reference
        item_features (scipy sparse CSR matrix): item feature matric
        model (LightFM instance): fitted LightFM model
        N (int): Number of top similar items to return

    Returns:
        pandas.DataFrame: top N most similar items with score
    """
    _, item_representations = model.get_item_representations(features=item_features)

    # Cosine similarity
    scores = item_representations.dot(item_representations[item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms

    best = np.argpartition(scores, -(N + 1))[-(N + 1) :]
    return pd.DataFrame(
        sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])[1:],
        columns=["itemID", "score"],
    )


def prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights):
    """Function to prepare test df for evaluation

    Args:
        test_idx (slice): slice of test indices
        uids (numpy.ndarray): Array of internal user indices
        iids (numpy.ndarray): Array of internal item indices
        uid_map (dict): Keys to map internal user indices to external ids.
        iid_map (dict): Keys to map internal item indices to external ids.
        weights (numpy.float32 coo_matrix): user-item interaction

    Returns:
        pandas.DataFrame: user-item selected for testing
    """
    test_df = pd.DataFrame(
        zip(
            uids[test_idx],
            iids[test_idx],
            [list(uid_map.keys())[x] for x in uids[test_idx]],
            [list(iid_map.keys())[x] for x in iids[test_idx]],
        ),
        columns=["uid", "iid", "userID", "itemID"],
    )

    dok_weights = weights.todok()
    test_df["rating"] = test_df.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)

    return test_df[["userID", "itemID", "rating"]]


def prepare_all_predictions(
    data,
    uid_map,
    iid_map,
    interactions,
    model,
    num_threads,
    user_features=None,
    item_features=None,
):
    """Function to prepare all predictions for evaluation.
    Args:
        data (pandas df): dataframe of all users, items and ratings as loaded
        uid_map (dict): Keys to map internal user indices to external ids.
        iid_map (dict): Keys to map internal item indices to external ids.
        interactions (np.float32 coo_matrix): user-item interaction
        model (LightFM instance): fitted LightFM model
        num_threads (int): number of parallel computation threads
        user_features (np.float32 csr_matrix): User weights over features
        item_features (np.float32 csr_matrix):  Item weights over features
    Returns:
        pandas.DataFrame: all predictions
    """
    users, items, preds = [], [], []  # noqa: F841
    item = list(data.itemID.unique())
    for user in data.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items})
    all_predictions["uid"] = all_predictions.userID.map(uid_map)
    all_predictions["iid"] = all_predictions.itemID.map(iid_map)

    dok_weights = interactions.todok()
    all_predictions["rating"] = all_predictions.apply(
        lambda x: dok_weights[x.uid, x.iid], axis=1
    )

    all_predictions = all_predictions[all_predictions.rating < 1].reset_index(drop=True)
    all_predictions = all_predictions.drop("rating", axis=1)

    all_predictions["prediction"] = all_predictions.apply(
        lambda x: model.predict(
            user_ids=np.array([x["uid"]], dtype=np.int32),
            item_ids=np.array([x["iid"]], dtype=np.int32),
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )[0],
        axis=1,
    )

    return all_predictions[["userID", "itemID", "prediction"]]


class LightFMModel:
    def __init__(
            self,
            no_components=10, loss="warp", learning_schedule="adagrad", learning_rate=0.05, user_alpha=0.0, item_alpha=0.0, epochs=10,
            col_user=constants.DEFAULT_USER_COL,
            col_item=constants.DEFAULT_ITEM_COL,
            col_rating=constants.DEFAULT_RATING_COL,
            col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
            col_prediction=constants.DEFAULT_PREDICTION_COL,
            col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
            col_isimpr=constants.DEFAULT_ISIMPR_COL
        ):
        """
        Initialize the LightFM model.

        Args:
            no_components (int): Number of latent factors.
            loss (str): Loss function to use ("warp", "bpr", "warp-kos", "logistic").
            learning_schedule (str): Learning rate schedule ("adagrad" or "adadelta").
            user_alpha (float): L2 regularization parameter for users.
            item_alpha (float): L2 regularization parameter for items.
            col_user (str): Name of the user column in DataFrames.
            col_item (str): Name of the item column in DataFrames.
            col_rating (str): Name of the rating column in DataFrames.
            col_timestamp (str): Name of the timestamp column in DataFrames.
            col_prediction (str): Name of the prediction column in DataFrames.
            col_imprgroup (str): Name of the impression group column in DataFrames.
            col_isimpr (str): Name of the is impression column in DataFrames.
        """
        self.model = LightFM(
            no_components=no_components, learning_schedule=learning_schedule, loss=loss,
            learning_rate=learning_rate, user_alpha=user_alpha, item_alpha=item_alpha
        )
        self.epochs = epochs

        self.dataset = Dataset()
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        self.col_imprgroup = col_imprgroup
        self.col_isimpr = col_isimpr

        self.interactions = None
        self.weights = None
        self.user_features = None
        self.item_features = None

    def fit(self, df):
        """Main fit method for LightFMModel.

        Note:
            Please make sure that `df` has no duplicates.

        Args:
            df (pandas.DataFrame): User item rating dataframe (without duplicates).
        """
        # Identify userIDs where all ratings are 0
        all_zero_users = df.groupby(self.col_user)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_users].copy()

        # Identify itemIDs where all ratings are 0
        all_zero_items = df.groupby(self.col_item)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_items].drop_duplicates()

        # Prepare user and item mappings
        users = df[self.col_user].unique().tolist()
        items = df[self.col_item].unique().tolist()

        self.dataset.fit(users, items)

        # Create interaction matrix
        interactions, _ = self.dataset.build_interactions([
            (row[self.col_user], row[self.col_item], row[self.col_rating])
            for _, row in df.iterrows()
        ])

        self.interactions = interactions
        self.model = self.model.fit(interactions, epochs=self.epochs, num_threads=4)

    @staticmethod
    def get_raytune_space(model_name: str = "lightfm"):
        default_space = {
            "no_components": tune.randint(10, 201),  # Number of latent factors
            "loss": tune.choice(["warp", "bpr", "logistic"]),  # Loss function
            "learning_schedule": tune.choice(["adadelta"]),  # Learning schedule (adagrad)
            "learning_rate": tune.loguniform(0.001, 0.01),  # Learning rate step
            "user_alpha": tune.loguniform(1e-6, 1e-2),  # Regularization for user factors
            "item_alpha": tune.loguniform(1e-6, 1e-2),  # Regularization for item factors
            "epochs": tune.randint(10, 51),  # Number of training epochs
        }

        return default_space

    def predict(self, test):
        """Output LightFM scores for only the users-items pairs which are in the test set.

        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test.

        Returns:
            pandas.DataFrame: DataFrame containing the prediction results.
        """
        # Map user and item IDs from dataset mappings
        user_ids = test[self.col_user].map(lambda x: self.dataset.mapping()[0].get(x, None))
        item_ids = test[self.col_item].map(lambda x: self.dataset.mapping()[1].get(x, None))

        # Identify valid pairs where user and item IDs are not None
        valid_pairs = ~user_ids.isna() & ~item_ids.isna()

        # Initialize the result with zeros for all predictions
        predictions = np.zeros(len(test))

        # Predict only for valid user-item pairs
        if valid_pairs.sum() > 0:
            if self.user_features is None:
                predictions[valid_pairs] = self.model.predict(
                    user_ids[valid_pairs].astype(int).values,
                    item_ids[valid_pairs].astype(int).values
                )
            else:
                if self.item_features is None:
                    predictions[valid_pairs] = self.model.predict(
                        user_ids[valid_pairs].astype(int).values,
                        item_ids[valid_pairs].astype(int).values,
                        user_features=self.user_features
                    )
                else:
                    predictions[valid_pairs] = self.model.predict(
                        user_ids[valid_pairs].astype(int).values,
                        item_ids[valid_pairs].astype(int).values,
                        item_features=self.item_features,
                        user_features=self.user_features
                    )

        predictions += abs(predictions.min())
        predictions /= predictions.max()

        # Create a result DataFrame
        res = test.copy()
        res[self.col_prediction] = predictions

        # Randomized noise for granting the same order between executions in case of ties
        def add_rand_noise(row, max_noise=1e-10):
            # Combine user and item IDs into a string and hash it to create a seed
            seed = int(f"{row[self.col_user]}{row[self.col_item]}")
            rnd = np.random.default_rng(seed)  # Create a temporary RNG with the specific seed
            return rnd.uniform(0, max_noise)  # Generate a random number in [0.0, max_noise)

        zero_ids = res[self.col_prediction] == 0.0
        if zero_ids.any():
            res.loc[zero_ids, self.col_prediction] = res[zero_ids].apply(add_rand_noise, axis=1)

        # Grant values over than randomization
        res[self.col_prediction] = np.where(
            ~zero_ids, res[self.col_prediction] + 0.02, res[self.col_prediction]
        )

        # Avoid negative scores
        res[self.col_prediction] += abs(res[self.col_prediction].min()) + 0.01

        # Avoid ties
        res[self.col_prediction] += res.apply(add_rand_noise, axis=1)

        return res

    def save_model(self, filepath):
        """Save the model and related sparse matrices to a file.
        Args:
            filepath (str): Path to save the model and data.
        """
        os.makedirs(filepath, exist_ok=True)

        # Save the LightFM model using pickle
        model_filename = os.path.join(filepath, "lightfm_model.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(self.model, f)

        # Save the dataset using pickle
        dataset_filename = os.path.join(filepath, "lightfm_dataset.pkl")
        with open(dataset_filename, "wb") as f:
            pickle.dump(self.dataset, f)

        # Save interactions, weights, user_features, and item_features only if they are not None
        if self.interactions is not None:
            interactions_filename = os.path.join(filepath, "interactions.npz")
            sparse.save_npz(interactions_filename, self.interactions)

        if self.weights is not None:
            weights_filename = os.path.join(filepath, "weights.npz")
            sparse.save_npz(weights_filename, self.weights)

        if self.user_features is not None:
            user_features_filename = os.path.join(filepath, "user_features.npz")
            sparse.save_npz(user_features_filename, self.user_features)

        if self.item_features is not None:
            item_features_filename = os.path.join(filepath, "item_features.npz")
            sparse.save_npz(item_features_filename, self.item_features)

    def load_model(self, filepath):
        """Load the model and related sparse matrices from a file.
        Args:
            filepath (str): Path to load the model and data from.
        Returns:
            bool: True if model successfully loaded, otherwise False.
        """
        try:
            # Load the LightFM model using pickle
            model_filename = os.path.join(filepath, "lightfm_model.pkl")
            with open(model_filename, "rb") as f:
                self.model = pickle.load(f)

            # Load the dataset using pickle
            dataset_filename = os.path.join(filepath, "lightfm_dataset.pkl")
            with open(dataset_filename, "rb") as f:
                self.dataset = pickle.load(f)

            # Load interactions, weights, user_features, and item_features if they exist
            interactions_filename = os.path.join(filepath, "interactions.npz")
            if os.path.exists(interactions_filename):
                self.interactions = sparse.load_npz(interactions_filename)

            weights_filename = os.path.join(filepath, "weights.npz")
            if os.path.exists(weights_filename):
                self.weights = sparse.load_npz(weights_filename)

            user_features_filename = os.path.join(filepath, "user_features.npz")
            if os.path.exists(user_features_filename):
                self.user_features = sparse.load_npz(user_features_filename)

            item_features_filename = os.path.join(filepath, "item_features.npz")
            if os.path.exists(item_features_filename):
                self.item_features = sparse.load_npz(item_features_filename)

            return True
        except Exception as ex:
            print(f"Failed to load model: {ex}")
            return False


class LightFMFModel(LightFMModel):
    def get_raytune_space(model_name: str = "lightfmf"):
        default_space = {
            "no_components": tune.randint(10, 201),  # Number of latent factors
            "loss": tune.choice(["warp"]),  # Loss function ["bpr", "logistic"]
            "learning_schedule": tune.choice(["adadelta"]),  # Learning schedule (adagrad)
            "learning_rate": tune.loguniform(0.001, 0.01),  # Learning rate step
            "user_alpha": tune.loguniform(1e-6, 1e-2),  # Regularization for user factors
            "item_alpha": tune.loguniform(1e-6, 1e-2),  # Regularization for item factors
            "epochs": tune.randint(10, 51),  # Number of training epochs
        }

        return default_space

    def discretize_by_quantiles(self, data, num_quantiles=4, labels=True):
        """
        Discretize continuous values into quantiles and label by min/max of group.

        Parameters:
        - data (pd.Series): The continuous values to discretize.
        - num_quantiles (int): Number of quantiles to divide the data into.
        - labels (bool): Whether to label groups by their min/max range. If False, labels by quantile number.

        Returns:
        - pd.Series: Discretized values with min/max labels or quantile numbers.
        """
        # Discretize data into quantile bins
        _, bin_edges = pd.qcut(data, q=num_quantiles, precision=2, retbins=True, duplicates='drop')

        if labels:
            # Create min/max labels for the bins
            bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]" for i in range(len(bin_edges) - 1)]
            return pd.cut(data, bins=bin_edges, labels=bin_labels, include_lowest=True)
        else:
            # Return bin numbers if labels=False
            return pd.qcut(data, q=num_quantiles, labels=False, precision=2, duplicates='drop')

    def transform_features(self, df, id_col):
        df = df.copy()

        for c in df.select_dtypes("float"):
            df[c] = self.discretize_by_quantiles(df[c])

        # df[id_col + "_diagonal"] = df[id_col]
        categorize_columns = list(df.columns)
        categorize_columns.remove(id_col)
        df[categorize_columns] = (
            df[categorize_columns].astype("str").radd(np.array(categorize_columns) + ":")
        )

        unique_feature_values = []
        for c in categorize_columns:
            unique_feature_values.extend(df[c].unique().tolist())

        return df[id_col].values.tolist(), unique_feature_values, df

    def fitf(self, df, user_features, item_features):
        """Main fit method for LightFMModel.

        Note:
            Please make sure that `df` has no duplicates.

        Args:
            df (pd.DataFrame): User item rating dataframe (without duplicates).
            user_features (pd.DataFrame): Features representing the users.
            item_features (pd.DataFrame): Features representing the items.
        """

        # Identify userIDs where all ratings are 0
        all_zero_users = df.groupby(self.col_user)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_users].copy()

        # Identify itemIDs where all ratings are 0
        all_zero_items = df.groupby(self.col_item)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_items].drop_duplicates()

        # Filter user and item features based on remaining user and item IDs in df
        filtered_user_ids = df[self.col_user].unique()
        filtered_item_ids = df[self.col_item].unique()

        # TODO: Allow only user or item features
        user_features = user_features[user_features[self.col_user].isin(filtered_user_ids)].copy()
        item_features = item_features[item_features[self.col_item].isin(filtered_item_ids)].copy()

        # Ensure there are no NaN or infinite values in user and item features
        user_features = user_features.replace([np.inf, -np.inf], np.nan).dropna()
        item_features = item_features.replace([np.inf, -np.inf], np.nan).dropna()

        # Prepare user and item features
        users, unique_user_features, user_features_dft = self.transform_features(
            user_features, self.col_user
        )
        items, unique_items_features, item_features_dft = self.transform_features(
            item_features, self.col_item
        )

        # Fit dataset
        self.dataset.fit(users, items, unique_user_features, unique_items_features)

        # Extract user and item feature arrays for faster processing
        user_ids = user_features_dft[self.col_user].values
        user_onlyfeatures = user_features_dft.drop(columns=[self.col_user]).values
        item_ids = item_features_dft[self.col_item].values
        item_onlyfeatures = item_features_dft.drop(columns=[self.col_item]).values

        # Build user and item features
        self.user_features = self.dataset.build_user_features(
            list(zip(user_ids, user_onlyfeatures.tolist()))
        )
        self.item_features = self.dataset.build_item_features(
            list(zip(item_ids, item_onlyfeatures.tolist()))
        )

        # Create interaction matrix
        self.interactions, self.weights = self.dataset.build_interactions(
            df[[self.col_user, self.col_item, self.col_rating]].values
        )

        self.model = self.model.fit(
            self.interactions, self.user_features, self.item_features,
            sample_weight=self.weights, epochs=self.epochs, num_threads=4,
        )


class LightFMFUModel(LightFMFModel):
    def fitf(self, df, user_features, item_features):
        """Main fit method for LightFMModel.

        Note:
            Please make sure that `df` has no duplicates.

        Args:
            df (pd.DataFrame): User item rating dataframe (without duplicates).
            user_features (pd.DataFrame): Features representing the users.
            item_features (pd.DataFrame): Features representing the items.
        """

        # Identify userIDs where all ratings are 0
        all_zero_users = df.groupby(self.col_user)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_users].copy()

        # Identify itemIDs where all ratings are 0
        all_zero_items = df.groupby(self.col_item)[self.col_rating].transform(lambda x: (x == 0).all())
        df = df[~all_zero_items].drop_duplicates()

        # Filter user and item features based on remaining user and item IDs in df
        filtered_user_ids = df[self.col_user].unique()
        filtered_item_ids = df[self.col_item].unique()

        # TODO: Allow only user or item features
        user_features = user_features[user_features[self.col_user].isin(filtered_user_ids)].copy()

        # Ensure there are no NaN or infinite values in user and item features
        user_features = user_features.replace([np.inf, -np.inf], np.nan).dropna()

        # Prepare user and item features
        users, unique_user_features, user_features_dft = self.transform_features(
            user_features, self.col_user
        )
        items, _, item_features_dft = self.transform_features(
            item_features, self.col_item
        )

        # Fit dataset
        self.dataset.fit(users, items, unique_user_features)

        # Extract user and item feature arrays for faster processing
        user_ids = user_features_dft[self.col_user].values
        user_onlyfeatures = user_features_dft.drop(columns=[self.col_user]).values
        item_ids = item_features_dft[self.col_item].values

        # Build user and item features
        self.user_features = self.dataset.build_user_features(
            list(zip(user_ids, user_onlyfeatures.tolist()))
        )
        self.item_features = None

        # Create interaction matrix
        self.interactions, self.weights = self.dataset.build_interactions(
            df[[self.col_user, self.col_item, self.col_rating]].values
        )

        self.model = self.model.fit(
            self.interactions, self.user_features,
            sample_weight=self.weights, epochs=self.epochs, num_threads=4,
        )
