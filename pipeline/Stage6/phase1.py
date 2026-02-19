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
from safetensors.tensorflow import save_file
import joblib


class Phase1:
    def __init__(self, prev_stage, data_path, train, val, test, feature_report, seed=0):
        self.seed = seed
        self._set_seed()

        data_dir = os.path.join(prev_stage, data_path)

        # Fix encoding error with explicit utf-8
        with open(os.path.join(data_dir, feature_report), 'r', encoding='utf-8') as f:
            self.features = json.load(f)

        self.train = pd.read_csv(os.path.join(data_dir, train), encoding='utf-8')
        self.val = pd.read_csv(os.path.join(data_dir, val), encoding='utf-8')
        self.test = pd.read_csv(os.path.join(data_dir, test), encoding='utf-8')

    def _set_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def get_features(self, part=1):
        """Selects features based on Part 1 (Clinical) or Part 2 (Labs added)."""
        feature_list = (
                self.features.get("Clinical", []) +
                self.features.get("Demographic / Other", []) +
                self.features.get("Scoring", []) +
                self.features.get("clinical_derived", []))

        if part == 2:
            feature_list += (self.features.get("Laboratory", []) +
                             self.features.get("lab_derived", []))

        available_features = [f for f in feature_list if f in self.train.columns]
        return available_features

    def prepare_data(self, part=1):
        target = "Diagnosis"
        feats = self.get_features(part)

        scaler = StandardScaler()
        # Scale only based on Training data to prevent data leakage
        x_train = scaler.fit_transform(self.train[feats])
        y_train = self.train[target]

        x_val = scaler.transform(self.val[feats])
        y_val = self.val[target]

        x_test = scaler.transform(self.test[feats])
        y_test = self.test[target]

        return x_train, x_val, x_test, y_train, y_val, y_test, feats, scaler

    def define_MLP(self, input_dim):
        """Your tested architecture: 32 -> 16 (Embedding) -> 1 (Output)."""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu', name="clinical_embedding"),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def save_as_safetensors(self, model, output_path):
        """Saves weights in .safetensors format for cross-platform compatibility."""
        tensors = {v.name: v.read_value() for v in model.weights}
        save_file(tensors, output_path)

    def extract_embeddings(self, model, x_data):
        """Returns the 16-D vector from the clinical_embedding layer."""
        embedding_model = Model(inputs=model.input, outputs=model.get_layer("clinical_embedding").output)
        return embedding_model.predict(x_data)

    def train_phase_part(self, part, threshold, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        part_name = f"phase1_part{part}"

        # 1. Prepare Data
        x_train, x_val, x_test, y_train, y_val, y_test, feats, scaler = self.prepare_data(part)

        # 2. Train
        model = self.define_MLP(len(feats))
        early_stop = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)

        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)

        # 3. Predict & Report using experimental threshold
        probs = model.predict(x_test).ravel()
        preds = (probs >= threshold).astype(int)

        report = classification_report(y_test, preds, output_dict=True)
        report['auc'] = roc_auc_score(y_test, probs)
        report['threshold'] = threshold

        # 4. Save Assets
        model.save(os.path.join(output_dir, f"{part_name}_model.h5"))
        self.save_as_safetensors(model, os.path.join(output_dir, f"{part_name}_model.safetensors"))
        joblib.dump(scaler, os.path.join(output_dir, f"{part_name}_scaler.pkl"))

        # Save Embeddings
        embeddings = {"train": x_train, "val": x_val, "test": x_test}
        for split_name, x_data in embeddings.items():
            emb = self.extract_embeddings(model, x_data)
            np.save(os.path.join(output_dir, f"{part_name}_{split_name}_embeddings.npy"), emb)

        with open(os.path.join(output_dir, f"{part_name}_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)

        return report