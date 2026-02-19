import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib


class Phase1:
    """
    Phase 1: Tabular Clinical Model
    Goal: Predict Diagnosis & produce Clinical Embeddings for fusion.
    """

    def __init__(self, data_dir, train_file, val_file, test_file,
                 feature_groups_file, seed=0, threshold=0.614):

        self.seed = seed
        self.threshold = threshold
        self._set_seed()

        # Load feature groups
        with open(os.path.join(data_dir, feature_groups_file), 'r', encoding='utf-8') as f:
            self.feature_groups = json.load(f)

        # Load splits
        self.train_df = pd.read_csv(os.path.join(data_dir, train_file))
        self.val_df = pd.read_csv(os.path.join(data_dir, val_file))
        self.test_df = pd.read_csv(os.path.join(data_dir, test_file))

        self.target = "Diagnosis"

    # ------------------------------------------------------------------
    # 🔒 Reproducibility
    # ------------------------------------------------------------------
    def _set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    # ------------------------------------------------------------------
    # 📊 Feature Selection (Final Phase 1 design)
    # ------------------------------------------------------------------
    def get_phase1_features(self):
        feature_list = (
            self.feature_groups["Clinical"] +
            self.feature_groups["Laboratory"] +
            self.feature_groups["Demographic / Other"] +
            self.feature_groups["Scoring"] +
            self.feature_groups["clinical_derived"] +
            self.feature_groups["lab_derived"]
        )

        available = [f for f in feature_list if f in self.train_df.columns]
        return available

    # ------------------------------------------------------------------
    # 🧹 Data Preparation
    # ------------------------------------------------------------------
    def prepare_data(self):
        features = self.get_phase1_features()

        scaler = StandardScaler()

        X_train = scaler.fit_transform(self.train_df[features])
        y_train = self.train_df[self.target]

        X_val = scaler.transform(self.val_df[features])
        y_val = self.val_df[self.target]

        X_test = scaler.transform(self.test_df[features])
        y_test = self.test_df[self.target]

        return X_train, X_val, X_test, y_train, y_val, y_test, features, scaler

    # ------------------------------------------------------------------
    # 🧠 Final Small MLP Architecture (validated)
    # ------------------------------------------------------------------
    def build_model(self, input_dim):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(16, activation="relu", name="clinical_embedding"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        return model

    # ------------------------------------------------------------------
    # 🧬 Embedding Extraction
    # ------------------------------------------------------------------
    def extract_embeddings(self, model, X):
        embedding_model = Model(
            inputs=model.input,
            outputs=model.get_layer("clinical_embedding").output
        )
        return embedding_model.predict(X)

    # ------------------------------------------------------------------
    # 💾 Save Artifacts
    # ------------------------------------------------------------------
    def save_assets(self, output_dir, model, scaler, features,
                    embeddings, report):

        os.makedirs(output_dir, exist_ok=True)

        # Model
        model.save(os.path.join(output_dir, "phase1_model.h5"))

        # Scaler
        joblib.dump(scaler, os.path.join(output_dir, "phase1_scaler.pkl"))

        # Features used
        with open(os.path.join(output_dir, "phase1_features.json"), "w") as f:
            json.dump(features, f, indent=4)

        # Embeddings
        for split, emb in embeddings.items():
            np.save(os.path.join(output_dir, f"{split}_clinical_embeddings.npy"), emb)

        # Report
        with open(os.path.join(output_dir, "phase1_report.json"), "w") as f:
            json.dump(report, f, indent=4)

    # ------------------------------------------------------------------
    # 🚀 Train + Evaluate + Save
    # ------------------------------------------------------------------
    def run(self, output_dir="phase1_output"):
        # 1️⃣ Prepare Data
        X_train, X_val, X_test, y_train, y_val, y_test, features, scaler = self.prepare_data()

        # 2️⃣ Build Model
        model = self.build_model(len(features))

        # 3️⃣ Train
        early_stop = EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=1
        )

        # 4️⃣ Predictions
        probs = model.predict(X_test).ravel()
        preds = (probs >= self.threshold).astype(int)

        # 5️⃣ Metrics
        report = classification_report(y_test, preds, output_dict=True)
        report["auc"] = float(roc_auc_score(y_test, probs))
        report["accuracy"] = float(accuracy_score(y_test, preds))
        report["threshold"] = self.threshold
        report["seed"] = self.seed

        # 6️⃣ Embeddings for Fusion
        embeddings = {
            "train": self.extract_embeddings(model, X_train),
            "val": self.extract_embeddings(model, X_val),
            "test": self.extract_embeddings(model, X_test)
        }

        # 7️⃣ Save Everything
        self.save_assets(output_dir, model, scaler, features, embeddings, report)

        return report
