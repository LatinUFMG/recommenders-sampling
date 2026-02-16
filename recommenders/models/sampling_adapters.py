import os
import json
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sparse
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hyperopt import hp
from hyperopt.pyll.base import scope
from ray import tune

from .sasrec.model import SASREC
from .sasrec.util import SASRecDataSet
from .sasrec.sampler import WarpSampler
from .vae.standard_vae import StandardVAE
from .ncf.ncf_singlenode import NCF
from .ncf.dataset import Dataset as NCFDataset
from .deeprec.models.graphrec.lightgcn import LightGCN
from .deeprec.DataModel.ImplicitCF import ImplicitCF
from fastai.collab import collab_learner, CollabDataLoaders
from fastai.callback.progress import CSVLogger
from ..utils import constants


def apply_prediction_noise(df, col_user, col_item, col_prediction):
    df[col_prediction] = df[col_prediction].astype(float)
    zero_mask = df[col_prediction] == 0.0

    if zero_mask.any():
        temp_ids = df.loc[zero_mask, [col_user, col_item]].fillna(0).astype(int)

        def _noise(row):
            seed = int(f"{row[col_user]}{row[col_item]}")
            return np.random.default_rng(seed).uniform(0.0, 0.01)

        noise_values = temp_ids.apply(_noise, axis=1)
        df.loc[zero_mask, col_prediction] = noise_values

    df[col_prediction] = np.where(~zero_mask, df[col_prediction] + 0.02, df[col_prediction])
    df[col_prediction] += abs(df[col_prediction].min()) + 0.01
    return df


# ==============================================================================
#  1. OPTIMIZED SASRec (High Performance)
# ==============================================================================
class SASRecPyTorch(nn.Module):
    def __init__(self, item_num, seq_max_len, num_blocks, embedding_dim, num_heads, dropout_rate, l2_reg):
        super().__init__()
        self.item_num = item_num
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg

        self.item_emb = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_max_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_seq):
        device = input_seq.device
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]

        key_padding_mask = (input_seq == 0)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

        seqs = self.item_emb(input_seq)
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.pos_emb(positions)

        seqs += pos_embs
        seqs = self.emb_dropout(seqs)
        seqs = self.layer_norm(seqs)

        features = self.transformer(seqs, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return features


class SASRecTrainDataset(Dataset):
    """
    Optimized Dataset that pre-computes sequences and uses vectorized negative sampling.
    """
    def __init__(self, user_train, itemnum, max_len):
        self.user_train = user_train
        self.users = sorted(list(user_train.keys()))
        self.itemnum = itemnum
        self.max_len = max_len

        # Pre-compute static data structure to save CPU during training
        # (user, input_seq, target_pos, set_of_items_for_neg_check)
        self.samples = []
        for u in self.users:
            seq = user_train[u]
            if len(seq) > max_len:
                # Use the last max_len interactions
                input_seq = seq[-max_len-1:-1]
                pos = seq[-max_len:]
            else:
                input_seq = seq[:-1]
                pos = seq[1:]

            pad_len = max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq
            pos = [0] * pad_len + pos

            self.samples.append((u, np.array(input_seq), np.array(pos), set(seq)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, input_seq, pos, seq_set = self.samples[idx]

        # Vectorized Negative Sampling (Much faster than simple loops)
        neg = self._rnd.randint(1, self.itemnum + 1, size=len(pos))

        # Fast collision check
        # For typical sparse datasets, collision rate is < 0.1%, so this loop rarely runs
        for i, val in enumerate(neg):
            while val in seq_set:
                val = self._rnd.randint(1, self.itemnum + 1)
                neg[i] = val

        return torch.tensor(u), torch.LongTensor(input_seq), torch.LongTensor(pos), torch.LongTensor(neg)


class SASRecAdapter:
    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
        col_isimpr=constants.DEFAULT_ISIMPR_COL,
        max_seq_length=50, hidden_units=100, batch_size=128, num_blocks=2, num_heads=1,
        dropout_rate=0.1, l2_emb=0.0, learning_rate=0.001, epochs=20, seed=42, *args, **kwargs
    ):
        self.col_user, self.col_item, self.col_rating = col_user, col_item, col_rating
        self.col_timestamp, self.col_prediction = col_timestamp, col_prediction
        self.col_imprgroup, self.col_isimpr = col_imprgroup, col_isimpr

        self.max_seq_length = max_seq_length
        self.hidden_units = hidden_units
        self.batch_size, self.num_blocks, self.num_heads = batch_size, num_blocks, num_heads
        self.dropout_rate, self.l2_emb, self.learning_rate = dropout_rate, l2_emb, learning_rate
        self.epochs, self.seed = epochs, seed

        self.model = None
        self.dataset = None
        self.index2user, self.index2item = None, None
        self.user2index, self.item2index = None, None

    @staticmethod
    def get_hyperopt_space(model_type="sasrec"):
        return {
            "hidden_units": scope.int(hp.choice("hidden_units", [50, 100, 200])),
            "num_blocks": scope.int(hp.choice("num_blocks", [1, 2, 3])),
            "num_heads": scope.int(hp.choice("num_heads", [1, 2, 4])),
            "learning_rate": hp.loguniform("learning_rate", -5, -2),
            "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
        }

    @staticmethod
    def get_raytune_space(model_type="sasrec"):
        return {
            "hidden_units": tune.choice([64, 128, 256]),
            "num_blocks": tune.choice([1, 2, 3]),
            "num_heads": tune.choice([1, 2, 4]),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "dropout_rate": tune.uniform(0.0, 0.5),
        }

    def fit(self, df):
        # 1. Indexing
        users = df[self.col_user].astype(str).unique()
        items = df[self.col_item].astype(str).unique()
        self.index2user = dict(enumerate(users, start=1))
        self.index2item = dict(enumerate(items, start=1))
        self.user2index = {v: k for k, v in self.index2user.items()}
        self.item2index = {v: k for k, v in self.index2item.items()}

        # 2. Data Preparation
        tmp_file = "__sasrec_train_temp.txt"
        data = df.copy()
        data[self.col_user] = data[self.col_user].astype(str).map(self.user2index)
        data[self.col_item] = data[self.col_item].astype(str).map(self.item2index)
        data = data.sort_values(self.col_timestamp)
        data[[self.col_user, self.col_item]].to_csv(tmp_file, sep=" ", index=False, header=False)

        # Use library utility just to parse the file into a dict (fast enough)
        self.dataset = SASRecDataSet(filename=tmp_file)
        self.dataset.split()

        self.user_train = self.dataset.user_train

        # 3. Create Fast DataLoader
        # We replace WarpSampler with a PyTorch DataLoader
        train_ds = SASRecTrainDataset(self.dataset.user_train, self.dataset.itemnum, self.max_seq_length)
        # num_workers=0 ensures safety within Ray/Multiprocessing, but shuffle=True ensures good mixing
        # You can try num_workers=2 if not running inside Ray Tune
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # 4. Model Init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.hidden_units % self.num_heads != 0:
            self.num_heads = 1

        self.model = SASRecPyTorch(
            item_num=self.dataset.itemnum, seq_max_len=self.max_seq_length,
            num_blocks=self.num_blocks, embedding_dim=self.hidden_units,
            num_heads=self.num_heads, dropout_rate=self.dropout_rate, l2_reg=self.l2_emb
        )
        self.model.to(device)
        self.model.train()

        # 5. Training Loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98))
        bce_loss = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            for batch in train_loader:
                # Unpack batch from DataLoader
                u, seq, pos, neg = batch
                seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)

                optimizer.zero_grad()
                features = self.model(seq)

                pos_embs = self.model.item_emb(pos)
                neg_embs = self.model.item_emb(neg)

                pos_logits = (features * pos_embs).sum(dim=-1)
                neg_logits = (features * neg_embs).sum(dim=-1)

                loss = bce_loss(pos_logits, torch.ones_like(pos_logits)) + \
                       bce_loss(neg_logits, torch.zeros_like(neg_logits))

                # L2 Reg
                for param in self.model.item_emb.parameters():
                    loss += self.l2_emb * torch.norm(param)

                loss.backward()
                optimizer.step()

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    def predict(self, test):
        df = test.copy()

        # 1. Map users/items to internal indices
        df[self.col_user] = df[self.col_user].astype(str).map(self.user2index)
        df[self.col_item] = df[self.col_item].astype(str).map(self.item2index)

        mask = df[self.col_user].notna() & df[self.col_item].notna()
        scores = np.zeros(len(df), dtype=np.float32)

        if mask.any():
            device = next(self.model.parameters()).device
            self.model.eval()

            valid_df = df[mask]
            user_ids = valid_df[self.col_user].astype(int).values
            item_ids = valid_df[self.col_item].astype(int).values
            write_idx = np.flatnonzero(mask)

            # 2. Build history map once per unique user
            unique_users = np.unique(user_ids)
            user_history_map = {}

            for u in unique_users:
                hist = self.user_train.get(u, [])
                if not isinstance(hist, (list, np.ndarray)):
                    hist = []
                # Safe truncation
                hist = hist[-self.max_seq_length:]
                user_history_map[u] = hist

            # 3. Construct padded sequences
            seqs = np.zeros((len(user_ids), self.max_seq_length), dtype=np.int64)

            for i, u in enumerate(user_ids):
                h = user_history_map.get(u, [])
                l = len(h)
                if l > 0:
                    seqs[i, -l:] = h

            # 4. Skip empty sequences (prevents NaNs)
            valid_seq_mask = np.any(seqs != 0, axis=1)

            if valid_seq_mask.any():
                seqs_valid = seqs[valid_seq_mask]
                items_valid = item_ids[valid_seq_mask]
                write_idx_valid = write_idx[valid_seq_mask]

                batch_size = self.batch_size * 2
                num_batches = (len(seqs_valid) + batch_size - 1) // batch_size
                all_preds = []

                with torch.no_grad():
                    for b in range(num_batches):
                        start = b * batch_size
                        end = start + batch_size

                        b_seq = torch.LongTensor(seqs_valid[start:end]).to(device)
                        b_items = torch.LongTensor(items_valid[start:end]).to(device)

                        features = self.model(b_seq)
                        final_feat = features[:, -1, :]
                        item_embs = self.model.item_emb(b_items)

                        logits = (final_feat * item_embs).sum(dim=-1)
                        all_preds.append(logits.cpu().numpy())

                scores[write_idx_valid] = np.concatenate(all_preds)

            # Empty-history users keep score = 0.0 (safe baseline)

        # 5. Finalize predictions
        df[self.col_prediction] = scores

        # Tie-safe noise injection
        df = apply_prediction_noise(
            df,
            self.col_user,
            self.col_item,
            self.col_prediction
        )

        # Restore original external IDs
        df[self.col_user] = test[self.col_user]
        df[self.col_item] = test[self.col_item]

        return df


    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "sasrec.pt"))
        with open(os.path.join(path, "sasrec_dataset.pkl"), "wb") as f:
            pickle.dump(self.dataset, f)
        with open(os.path.join(path, "sasrec_meta.json"), "w") as f:
            json.dump({
                "user_train": self.user_train,
                "index2user": self.index2user, "index2item": self.index2item,
                "max_seq_length": self.max_seq_length, "hidden_units": self.hidden_units,
                "num_blocks": self.num_blocks, "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate, "l2_emb": self.l2_emb
            }, f)

    def load_model(self, path):
        try:
            with open(os.path.join(path, "sasrec_meta.json")) as f:
                meta = json.load(f)
            self.user_train = meta["user_train"]
            self.index2user = meta["index2user"]
            self.index2item = meta["index2item"]
            self.user2index = {v: int(k) for k, v in self.index2user.items()}
            self.item2index = {v: int(k) for k, v in self.index2item.items()}

            self.max_seq_length = meta["max_seq_length"]
            self.hidden_units = meta["hidden_units"]
            self.num_blocks, self.num_heads = meta["num_blocks"], meta["num_heads"]
            self.dropout_rate, self.l2_emb = meta.get("dropout_rate", 0.1), meta.get("l2_emb", 0.0)

            self.model = SASRecPyTorch(
                item_num=len(self.index2item), seq_max_len=self.max_seq_length,
                num_blocks=self.num_blocks, embedding_dim=self.hidden_units,
                num_heads=self.num_heads, dropout_rate=self.dropout_rate, l2_reg=self.l2_emb
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.load_state_dict(torch.load(os.path.join(path, "sasrec.pt"), map_location=device))

            with open(os.path.join(path, "sasrec_dataset.pkl"), "rb") as f:
                self.dataset = pickle.load(f)

            self.model.to(device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading SASRec: {e}")
            return False


# ==============================================================================
#  2. LightGCN Adapter
# ==============================================================================
class LightGCNAdapter:
    def __init__(self, col_user=constants.DEFAULT_USER_COL, col_item=constants.DEFAULT_ITEM_COL,
                 col_rating=constants.DEFAULT_RATING_COL, col_prediction=constants.DEFAULT_PREDICTION_COL,
                 col_imprgroup=constants.DEFAULT_IMPRGROUP_COL, col_isimpr=constants.DEFAULT_ISIMPR_COL,
                 embed_size=64, n_layers=3, batch_size=1024, decay=0.0001, epochs=10, learning_rate=0.005,
                 eval_epoch=5, top_k=10, seed=42, *args, **kwargs):
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.col_imprgroup = col_imprgroup
        self.col_isimpr = col_isimpr

        # Create hparams object
        self.hparams = type('', (), {})()
        self.hparams.embed_size = embed_size
        self.hparams.n_layers = n_layers
        self.hparams.batch_size = batch_size
        self.hparams.decay = decay
        self.hparams.epochs = epochs
        self.hparams.learning_rate = learning_rate
        self.hparams.eval_epoch = 10**9   # never trigger eval
        self.hparams.top_k = 0
        self.hparams.metrics = []

        # FIXED: Added 'save_model' attribute which was missing
        # The LightGCN constructor requires this to be set (True/False)
        self.hparams.save_model = False

        # These are used if save_model=True, but good to have to prevent other errors
        self.hparams.save_model_file = "lightgcn_model"
        self.hparams.save_epoch = eval_epoch
        self.hparams.metrics = ["recall", "ndcg", "map"]
        self.hparams.MODEL_DIR = "."

        self.seed = seed
        self._rnd = np.random.RandomState(self.seed)

        self.user2index = None
        self.item2index = None
        self.index2user = None
        self.index2item = None
        self.model = None

    def set_index(self, df):
        users = df[self.col_user].astype(str).unique()
        items = df[self.col_item].astype(str).unique()
        self.index2user = dict(enumerate(users))
        self.index2item = dict(enumerate(items))
        self.user2index = {v: k for k, v in self.index2user.items()}
        self.item2index = {v: k for k, v in self.index2item.items()}

    @staticmethod
    def get_hyperopt_space(model_type="lightgcn"):
        return {
            "n_layers": scope.int(hp.quniform("n_layers", 1, 4, 1)),
            "embed_size": scope.int(hp.choice("embed_size", [16, 32, 64, 128])),
            "learning_rate": hp.loguniform("learning_rate", -5, -2),
            "decay": hp.loguniform("decay", -6, -3),
            "batch_size": scope.int(hp.choice("batch_size", [512, 1024, 2048]))
        }

    @staticmethod
    def get_raytune_space(model_type="lightgcn"):
        return {
            "n_layers": tune.randint(1, 5),
            "embed_size": tune.choice([16, 32, 64, 128]),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "decay": tune.loguniform(1e-6, 1e-3),
            "batch_size": tune.choice([512, 1024, 2048]),
            "epochs": tune.randint(2, 11)
        }

    def fit(self, df: pd.DataFrame):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        # 1. Rebuild indices every fit (MANDATORY under CV / Ray)
        self.set_index(df)

        # 2. GPU config (Ray-safe)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        # 3. Map to internal IDs (let ImplicitCF handle reindexing)
        train_df = df.copy()
        train_df[self.col_user] = train_df[self.col_user].astype(str)
        train_df[self.col_item] = train_df[self.col_item].astype(str)

        # 4. Build ImplicitCF
        data = ImplicitCF(
            train=train_df,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            seed=self.seed
        )
        self.user2index = data.user2id
        self.item2index = data.item2id
        self.index2user = {v: k for k, v in self.user2index.items()}
        self.index2item = {v: k for k, v in self.item2index.items()}

        orig_train_loader = data.train_loader

        def safe_train_loader(batch_size):
            # Users WITH at least one interaction
            valid_df = data.interact_status[
                data.interact_status[self.col_item + "_interacted"]
                .apply(len) > 0
            ]

            if len(valid_df) == 0:
                raise RuntimeError("No users with interactions in this fold")

            valid_users = valid_df[self.col_user].values

            # Always sample with replacement (safer for CV folds)
            users = self._rnd.choice(valid_users, batch_size, replace=True)

            pos_items = []
            neg_items = []

            for u in users:
                interacted = valid_df.loc[
                    valid_df[self.col_user] == u,
                    self.col_item + "_interacted"
                ].values[0]

                # Positive
                pos_items.append(self._rnd.choice(list(interacted)))

                # Negative
                while True:
                    neg = self._rnd.randint(0, data.n_items - 1)
                    if neg not in interacted:
                        neg_items.append(neg)
                        break

            return (
                users.astype(np.int64),
                np.array(pos_items, dtype=np.int64),
                np.array(neg_items, dtype=np.int64),
            )


        data.train_loader = safe_train_loader

        # 5. Train model
        self.model = LightGCN(self.hparams, data, seed=self.seed)
        self.model.run_eval = lambda *args, **kwargs: None
        self.model.fit()
        self.train_df = df.copy()

    def predict(self, test: pd.DataFrame):
        df = test.copy()

        # Map external IDs to internal IDs used by ImplicitCF
        df[self.col_user] = df[self.col_user].astype(str).map(self.user2index)
        df[self.col_item] = df[self.col_item].astype(str).map(self.item2index)

        mask = df[self.col_user].notna() & df[self.col_item].notna()
        scores = np.zeros(len(df), dtype=np.float32)

        if mask.any():
            valid_df = df[mask]

            user_ids = valid_df[self.col_user].astype(int).values
            item_ids = valid_df[self.col_item].astype(int).values

            # Unique users to batch score
            uniq_users, inv = np.unique(user_ids, return_inverse=True)

            # Score ALL items for each user (TF graph)
            all_scores = self.model.score(
                uniq_users,
                remove_seen=False
            )  # shape: (n_users, n_items)

            # Pick scores for target (user, item) pairs
            preds = all_scores[inv, item_ids]
            scores[mask.values] = preds

        df[self.col_prediction] = scores
        df = apply_prediction_noise(df, self.col_user, self.col_item, self.col_prediction)

        # Restore original IDs
        df[self.col_user] = test[self.col_user]
        df[self.col_item] = test[self.col_item]

        return df

    def save_model(self, path):
        import tensorflow as tf
        os.makedirs(path, exist_ok=True)

        # -----------------------------------------
        # 1. Save TF checkpoint
        # -----------------------------------------
        ckpt_path = os.path.join(path, "lightgcn.ckpt")
        saver = tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.trainable_variables()
        )
        saver.save(self.model.sess, ckpt_path)


        # -----------------------------------------
        # 2. Save adapter metadata
        # -----------------------------------------
        with open(os.path.join(path, "lightgcn_attr.json"), "w") as f:
            json.dump(
                {
                    "index2user": self.index2user,
                    "index2item": self.index2item,
                    "embed_size": self.hparams.embed_size,
                    "n_layers": self.hparams.n_layers,
                    "decay": self.hparams.decay,
                },
                f,
            )

        # -----------------------------------------
        # 3. Save training dataframe
        # -----------------------------------------
        train_path = os.path.join(path, "lightgcn_train.pkl")
        with open(train_path, "wb") as f:
            pickle.dump(self.train_df, f)

    def load_model(self, path):
        try:
            import tensorflow as tf

            tf.compat.v1.reset_default_graph()
            # -----------------------------------------
            # 1. Load metadata
            # -----------------------------------------
            with open(os.path.join(path, "lightgcn_attr.json"), "r") as f:
                attr = json.load(f)

            self.index2user = attr["index2user"]
            self.index2item = attr["index2item"]
            self.user2index = {v: int(k) for k, v in self.index2user.items()}
            self.item2index = {v: int(k) for k, v in self.index2item.items()}

            # -----------------------------------------
            # 2. Load original training dataframe
            # -----------------------------------------
            with open(os.path.join(path, "lightgcn_train.pkl"), "rb") as f:
                self.train_df = pickle.load(f)

            # -----------------------------------------
            # 3. Rebuild ImplicitCF with REAL data
            # -----------------------------------------
            data = ImplicitCF(
                train=self.train_df,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_rating,
                seed=self.seed
            )

            # -----------------------------------------
            # 4. Rebuild LightGCN graph
            # -----------------------------------------
            self.model = LightGCN(self.hparams, data, seed=self.seed)

            # -----------------------------------------
            # 5. Restore checkpoint
            # -----------------------------------------
            ckpt_path = os.path.join(path, "lightgcn.ckpt")
            saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.trainable_variables()
            )
            saver.restore(self.model.sess, ckpt_path)

            return True

        except Exception as e:
            print(f"LightGCN load error: {e}")
            return False


# ==============================================================================
#  3. FastAI Adapter
# ==============================================================================
class FastAIAdapter:
    def __init__(self, col_user=constants.DEFAULT_USER_COL, col_item=constants.DEFAULT_ITEM_COL,
                 col_rating=constants.DEFAULT_RATING_COL, col_prediction=constants.DEFAULT_PREDICTION_COL,
                 col_imprgroup=constants.DEFAULT_IMPRGROUP_COL, col_isimpr=constants.DEFAULT_ISIMPR_COL,
                 n_factors=40, y_range=[0, 1.5], wd=0.1, epochs=5, lr=5e-3, bs=4096, seed=42, *args, **kwargs):
        self.col_user, self.col_item, self.col_rating = col_user, col_item, col_rating
        self.col_prediction = col_prediction
        self.col_imprgroup, self.col_isimpr = col_imprgroup, col_isimpr
        self.n_factors, self.y_range, self.wd = n_factors, y_range, wd
        self.epochs, self.lr, self.bs = epochs, lr, bs
        self.seed = seed
        self.user2index, self.item2index = None, None
        self.model = None

    def set_index(self, df):
        # Kept for API consistency, though FastAI handles its own mapping internally
        self.index2item = dict(enumerate(df[self.col_item].unique().astype("str").tolist()))
        self.index2user = dict(enumerate(df[self.col_user].unique().astype("str").tolist()))

    @staticmethod
    def get_hyperopt_space(model_type="fastai"):
        return {
            "n_factors": scope.int(hp.choice("n_factors", [20, 40, 80])),
            "lr": hp.loguniform("lr", -4, -1),
            "wd": hp.loguniform("wd", -5, -1)
        }

    @staticmethod
    def get_raytune_space(model_type="fastai"):
        return {
            "n_factors": tune.choice([16, 32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),
            "epochs": tune.randint(2, 11)
        }

    def fit(self, df):
        if self.item2index is None: self.set_index(df)

        # FastAI handles categorization, but we ensure string types for safety
        train_df = df.copy()
        train_df[self.col_user] = train_df[self.col_user].astype(str)
        train_df[self.col_item] = train_df[self.col_item].astype(str)

        dls = CollabDataLoaders.from_df(
            train_df,
            user_name=self.col_user,
            item_name=self.col_item,
            rating_name=self.col_rating if self.col_rating in df.columns else None,
            valid_pct=0.0,
            bs=self.bs,
            seed=self.seed
        )

        self.model = collab_learner(dls, n_factors=self.n_factors, y_range=self.y_range, wd=self.wd)
        self.model.fit_one_cycle(self.epochs, self.lr, cbs=[CSVLogger()])

    def predict(self, test):
        # 1. Keep original df untouched
        orig_df = test.copy()
        test_df = test.copy()

        # 2. Convert ONLY for fastai inference
        test_df[self.col_user] = test_df[self.col_user].astype(str)
        test_df[self.col_item] = test_df[self.col_item].astype(str)

        if self.col_rating in test_df.columns:
            test_df[self.col_rating] = test_df[self.col_rating].astype(float)

        # 3. Build test dataloader
        test_dl = self.model.dls.test_dl(
            test_df,
            with_labels=False
        )

        # 4. Predict
        preds = self.model.get_preds(dl=test_dl)[0]
        preds = preds.detach().cpu().numpy().flatten().astype("float32")

        # 5. Attach predictions to ORIGINAL df
        result_df = orig_df.copy()
        result_df[self.col_prediction] = preds

        # Ensure original dtypes preserved (Arrow-safe)
        result_df[self.col_user] = orig_df[self.col_user].astype(orig_df[self.col_user].dtype)
        result_df[self.col_item] = orig_df[self.col_item].astype(orig_df[self.col_item].dtype)

        # 6. Tie-breaking noise
        result_df = apply_prediction_noise(
            result_df,
            self.col_user,
            self.col_item,
            self.col_prediction
        )

        return result_df

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.export(os.path.join(path, "fastai_model.pkl"))

    def load_model(self, path):
        try:
            from fastai.learner import load_learner
            self.model = load_learner(os.path.join(path, "fastai_model.pkl"))
            return True
        except Exception as e:
            print(f"FastAI load error: {e}")
            return False


# ==============================================================================
#  4. EASE Adapter
# ==============================================================================
class EASEAdapter:
    """
    EASE^R: Embarrassingly Shallow Autoencoders for Sparse Data.
    A linear model that solves the weights analytically (Closed-Form).
    """
    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        col_imprgroup=constants.DEFAULT_IMPRGROUP_COL,
        col_isimpr=constants.DEFAULT_ISIMPR_COL,
        l2_norm=500.0,
        seed=42,
        *args, **kwargs
    ):
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.col_imprgroup = col_imprgroup
        self.col_isimpr = col_isimpr
        self.l2_norm = l2_norm
        self.seed = seed

        self.user2index = None
        self.item2index = None
        self.index2user = None
        self.index2item = None

        self.B = None
        self.X_train = None

    def set_index(self, df):
        users = df[self.col_user].astype(str).unique()
        items = df[self.col_item].astype(str).unique()
        self.index2user = dict(enumerate(users))
        self.index2item = dict(enumerate(items))
        self.user2index = {v: k for k, v in self.index2user.items()}
        self.item2index = {v: k for k, v in self.index2item.items()}
        self.n_users = len(self.index2user)
        self.n_items = len(self.index2item)

    @staticmethod
    def get_hyperopt_space(model_type="ease"):
        return {
            "l2_norm": scope.int(hp.quniform("l2_norm", 100, 1000, 50))
        }

    @staticmethod
    def get_raytune_space(model_type="ease"):
        return {
            "l2_norm": tune.randint(100, 1001)
        }

    def fit(self, df: pd.DataFrame):
        if self.item2index is None:
            self.set_index(df)

        # 1. Map Data to Sparse Matrix
        # Ensure indices are strictly integers
        rows = df[self.col_user].astype(str).map(self.user2index).astype(np.int32)
        cols = df[self.col_item].astype(str).map(self.item2index).astype(np.int32)

        if self.col_rating in df.columns:
             values = df[self.col_rating].values.astype(np.float32)
        else:
             values = np.ones(len(rows), dtype=np.float32)

        X = sparse.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))
        self.X_train = X

        # 2. Compute Gram Matrix (G = X^T * X)
        G = X.T.dot(X).toarray()

        # 3. Add Regularization (L2)
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.l2_norm

        # 4. Invert G (P = G^-1)
        try:
            P = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(G)

        # 5. Calculate Weight Matrix B
        # Original code: P / (-diag_P[:, None]) -> Divides by Row Diagonal (Wrong)
        # Correct EASE:  P / (-diag_P)          -> Divides by Column Diagonal
        diag_P = np.diag(P)

        # We divide column j by P_jj. Numpy broadcasts 1D array across columns.
        self.B = P / (-diag_P)
        self.B[diag_indices] = 0.0

        del G, P

    def predict(self, test):
        df = test.copy()
        df[self.col_user] = df[self.col_user].astype(str).map(self.user2index)
        df[self.col_item] = df[self.col_item].astype(str).map(self.item2index)

        mask = df[self.col_user].notna() & df[self.col_item].notna()
        scores = np.zeros(len(df), dtype=np.float32)

        if mask.any():
            valid_df = df[mask]

            # Get integer indices for valid rows only
            target_users = valid_df[self.col_user].astype(int).values
            target_items = valid_df[self.col_item].astype(int).values

            # Store indices in the original dataframe where we need to write results
            valid_write_indices = np.flatnonzero(mask)

            # Strategy: Compute scores batch by batch
            batch_size = 2048
            for start in range(0, len(target_users), batch_size):
                end = start + batch_size

                # Indices for this batch
                u_batch = target_users[start:end]
                i_batch = target_items[start:end]

                # Global indices for writing to 'scores'
                write_idx = valid_write_indices[start:end]

                # Get history: (batch, n_items)
                # Convert to dense for efficient einsum
                hist_batch = self.X_train[u_batch].toarray()

                # Get weights: (n_items, batch)
                # We need column i_batch[k] for user u_batch[k]
                weights = self.B[:, i_batch]

                # Score[k] = hist_batch[k] . weights[:, k]
                # einsum 'bi,ib->b':
                # b=batch dim, i=item dim.
                batch_scores = np.einsum('bi,ib->b', hist_batch, weights)

                # Write to specific indices in the original array
                scores[write_idx] = batch_scores

        df[self.col_prediction] = scores
        df = apply_prediction_noise(df, self.col_user, self.col_item, self.col_prediction)

        df[self.col_user] = test[self.col_user]
        df[self.col_item] = test[self.col_item]
        return df

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "ease_B.npy"), self.B)
        sparse.save_npz(os.path.join(path, "ease_X.npz"), self.X_train)
        with open(os.path.join(path, "ease_attr.json"), "w") as f:
            json.dump({"index2user": self.index2user, "index2item": self.index2item}, f)

    def load_model(self, path):
        try:
            self.B = np.load(os.path.join(path, "ease_B.npy"))
            self.X_train = sparse.load_npz(os.path.join(path, "ease_X.npz"))
            with open(os.path.join(path, "ease_attr.json"), "r") as f:
                attr = json.load(f)
                self.index2user = attr["index2user"]
                self.index2item = attr["index2item"]
                self.user2index = {v: k for k, v in self.index2user.items()}
                self.item2index = {v: k for k, v in self.index2item.items()}
                self.n_users = len(self.index2user)
                self.n_items = len(self.index2item)
            return True
        except: return False
