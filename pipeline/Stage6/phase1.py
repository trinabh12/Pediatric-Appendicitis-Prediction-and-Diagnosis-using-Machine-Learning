import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from safetensors.tensorflow import save_file


def build_clinical_mlp(input_dim):
    inputs = Input(shape=(input_dim,), name="tabular_input")

    # First Block
    x = Dense(32, activation="relu",
              kernel_regularizer=regularizers.l2(0.001),
              name="dense_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)

    # Second Block (Clinical Brain bottleneck)
    embedding_layer = Dense(16, activation="relu",
                            kernel_regularizer=regularizers.l2(0.001),
                            name="clinical_embedding")(x)
    x_emb = BatchNormalization(name="batch_norm_2")(embedding_layer)

    # Output
    outputs = Dense(1, activation="sigmoid", name="diagnosis_output")(x_emb)

    model = Model(inputs=inputs, outputs=outputs, name="Phase1_MLP")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model



class Phase1:
    def __init__(self, prev_stage, data_dir, train_file, val_file, test_file, feature_groups_file, seed, threshold):
        self.tabular_path = os.path.join(prev_stage, data_dir, "tabular")
        self.seed = seed
        self.threshold = threshold
        self._set_seed()

        with open(os.path.join(self.tabular_path, feature_groups_file), 'r', encoding='utf-8') as f:
            self.feature_groups = json.load(f)

        self.train_df = pd.read_csv(os.path.join(self.tabular_path, train_file), encoding='utf-8')
        self.val_df = pd.read_csv(os.path.join(self.tabular_path, val_file), encoding='utf-8')
        self.test_df = pd.read_csv(os.path.join(self.tabular_path, test_file), encoding='utf-8')
        self.target = "Diagnosis"

    def _set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def get_phase1_features(self, part=2):
        features = (self.feature_groups["Clinical"] + self.feature_groups["Demographic / Other"] +
                    self.feature_groups["Scoring"] + self.feature_groups["clinical_derived"])
        if part == 2:
            features += (self.feature_groups["Laboratory"] + self.feature_groups["lab_derived"])
        return [f for f in features if f in self.train_df.columns]

    def save_safetensors(self, model, output_path):
        weights = model.get_weights()
        weight_names = [v.name for v in model.weights]
        tensors = {name: tf.convert_to_tensor(weight) for name, weight in zip(weight_names, weights)}
        save_file(tensors, output_path)

    def _get_metrics(self, model, X, y, split_name):
        probs = model.predict(X, verbose=0).ravel()
        preds = (probs >= self.threshold).astype(int)
        acc = float(accuracy_score(y, preds))
        auc = float(roc_auc_score(y, probs))
        return {
            f"{split_name}_accuracy": acc,
            f"{split_name}_auc": auc
        }

    def train_phase_part(self, part, output_dir):
        features = self.get_phase1_features(part=part)

        y_train = self.train_df[self.target].values
        y_val = self.val_df[self.target].values
        y_test = self.test_df[self.target].values

        # 1. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.train_df[features])
        X_val_scaled = scaler.transform(self.val_df[features])
        X_test_scaled = scaler.transform(self.test_df[features])

        # 2. Feature Selection (Top 30 as per your notebook)
        k_features = min(30, len(features))
        selector = SelectKBest(score_func=f_classif, k=k_features)

        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_val_sel = selector.transform(X_val_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        # 3. Compute Balanced Class Weights
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

        # 4. Model Construction & Training
        model = build_clinical_mlp(input_dim=X_train_sel.shape[1])
        early_stop = EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True)

        print(f"--- Training Part {part} ({k_features} selected features) with L2 & BatchNorm ---")
        model.fit(
            X_train_sel, y_train,
            validation_data=(X_val_sel, y_val),
            epochs=100,
            batch_size=16,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        # 5. Generate Comprehensive Metrics Report
        report = {"threshold_used": self.threshold, "initial_features": len(features), "selected_features": k_features}
        report.update(self._get_metrics(model, X_train_sel, y_train, "train"))
        report.update(self._get_metrics(model, X_val_sel, y_val, "val"))
        report.update(self._get_metrics(model, X_test_sel, y_test, "test"))

        test_probs = model.predict(X_test_sel, verbose=0).ravel()
        test_preds = (test_probs >= self.threshold).astype(int)
        report["test_classification_report"] = classification_report(y_test, test_preds, output_dict=True)

        # 6. Save Model and Artifacts
        part_dir = os.path.join(output_dir, f"part_{part}")
        os.makedirs(part_dir, exist_ok=True)

        model.save(os.path.join(part_dir, "model.keras"))
        self.save_safetensors(model, os.path.join(part_dir, "model.safetensors"))
        joblib.dump(scaler, os.path.join(part_dir, "scaler.pkl"))
        joblib.dump(selector, os.path.join(part_dir, "feature_selector.pkl"))  # Save selector for Phase 3!

        # 7. Extract and Save 16-D Embeddings for Phase 3
        emb_model = Model(inputs=model.input, outputs=model.get_layer("clinical_embedding").output)
        np.save(os.path.join(part_dir, "train_embeddings.npy"), emb_model.predict(X_train_sel, verbose=0))
        np.save(os.path.join(part_dir, "val_embeddings.npy"), emb_model.predict(X_val_sel, verbose=0))
        np.save(os.path.join(part_dir, "test_embeddings.npy"), emb_model.predict(X_test_sel, verbose=0))

        with open(os.path.join(part_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=4)

        return report