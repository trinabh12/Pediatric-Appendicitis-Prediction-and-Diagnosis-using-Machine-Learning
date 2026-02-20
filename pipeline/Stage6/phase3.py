import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from safetensors.tensorflow import save_file


# ------------------------------------------------------------------
# 🧠 Phase 3 Architecture (Multi-Task Fusion Network)
# ------------------------------------------------------------------
def build_fusion_network(clinical_dim=16, image_dim=16, num_secondary_targets=6):
    # Inputs
    input_clinical = Input(shape=(clinical_dim,), name="clinical_embedding")
    input_image = Input(shape=(image_dim,), name="image_embedding")

    # Fusion (Concatenate the two 16-D vectors into a 32-D vector)
    merged = Concatenate()([input_clinical, input_image])

    # Shared Trunk (Learning how bloodwork and images interact)
    x = Dense(32, activation="relu", name="shared_dense_1")(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation="relu", name="shared_dense_2")(x)

    # -------------------------------------------------------------
    # 🎯 Head 1: Primary Diagnosis (Appendicitis Yes/No)
    # -------------------------------------------------------------
    primary_out = Dense(1, activation="sigmoid", name="primary_diagnosis")(x)

    # -------------------------------------------------------------
    # 🎯 Head 2: Secondary Findings (Probabilities of symptoms)
    # -------------------------------------------------------------
    secondary_out = Dense(num_secondary_targets, activation="sigmoid", name="secondary_findings")(x)

    # Compile Multi-Task Model
    model = Model(inputs=[input_clinical, input_image], outputs=[primary_out, secondary_out], name="Phase3_Fusion")

    # We weight the primary loss higher so the model prioritizes getting Appendicitis right
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "primary_diagnosis": "binary_crossentropy",
            "secondary_findings": "binary_crossentropy"
        },
        loss_weights={
            "primary_diagnosis": 1.0,
            "secondary_findings": 0.5
        },
        metrics={
            "primary_diagnosis": [tf.keras.metrics.AUC(name="auc"), "accuracy"],
            "secondary_findings": ["accuracy"]
        }
    )
    return model


# ------------------------------------------------------------------
# ⚙️ Phase 3 Manager
# ------------------------------------------------------------------
class Phase3:
    def __init__(self, tabular_dir, phase1_dir, phase2_dir, seed=0):
        self.tabular_dir = tabular_dir
        self.phase1_dir = phase1_dir
        self.phase2_dir = phase2_dir

        # Define the clinically relevant secondary targets we want to predict
        self.secondary_features = [
            'Free_Fluids',
            'Target_Sign',
            'Bowel_Wall_Thickening',
            'Pathological_Lymph_Nodes',
            'Appendicolith',
            'Perforation'
        ]

        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _load_data(self, split_name):
        """Loads embeddings and cleans target labels."""
        # Load Embeddings
        X_clin = np.load(os.path.join(self.phase1_dir, f"{split_name}_embeddings.npy"))
        X_img = np.load(os.path.join(self.phase2_dir, f"{split_name}_image_embeddings.npy"))

        # Load CSV for Targets
        df = pd.read_csv(os.path.join(self.tabular_dir, f"{split_name}_split.csv"))
        y_primary = df['Diagnosis'].values

        # Clean secondary labels: Map -1 (Not evaluated) to 0, and anything > 0 to 1
        y_secondary = df[self.secondary_features].copy()
        for col in self.secondary_features:
            y_secondary[col] = y_secondary[col].apply(lambda x: 1 if x > 0 else 0)
        y_secondary = y_secondary.values

        return X_clin, X_img, y_primary, y_secondary

    def save_safetensors(self, model, output_path):
        weights = model.get_weights()
        weight_names = [v.name for v in model.weights]
        tensors = {name: tf.convert_to_tensor(weight) for name, weight in zip(weight_names, weights)}
        save_file(tensors, output_path)

    def train_fusion(self, output_dir):
        print("Loading Phase 1 & Phase 2 Embeddings...")
        X_clin_train, X_img_train, y_p_train, y_s_train = self._load_data("train")
        X_clin_val, X_img_val, y_p_val, y_s_val = self._load_data("val")
        X_clin_test, X_img_test, y_p_test, y_s_test = self._load_data("test")

        model = build_fusion_network(num_secondary_targets=len(self.secondary_features))

        early_stop = EarlyStopping(monitor="val_primary_diagnosis_auc", patience=10, mode="max",
                                   restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_primary_diagnosis_auc", factor=0.5, patience=3, mode="max")

        print("--- Training Phase 3 Fusion Network ---")
        model.fit(
            x={"clinical_embedding": X_clin_train, "image_embedding": X_img_train},
            y={"primary_diagnosis": y_p_train, "secondary_findings": y_s_train},
            validation_data=(
                {"clinical_embedding": X_clin_val, "image_embedding": X_img_val},
                {"primary_diagnosis": y_p_val, "secondary_findings": y_s_val}
            ),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop, lr_scheduler],
            verbose=0
        )

        print("\nEvaluating Final Performance across all splits...")

        # 1. Train Metrics
        train_preds = model.predict({"clinical_embedding": X_clin_train, "image_embedding": X_img_train}, verbose=0)
        train_probs = train_preds[0].ravel()
        train_auc = float(roc_auc_score(y_p_train, train_probs))
        train_acc = float(accuracy_score(y_p_train, (train_probs >= 0.5).astype(int)))

        # 2. Validation Metrics
        val_preds = model.predict({"clinical_embedding": X_clin_val, "image_embedding": X_img_val}, verbose=0)
        val_probs = val_preds[0].ravel()
        val_auc = float(roc_auc_score(y_p_val, val_probs))
        val_acc = float(accuracy_score(y_p_val, (val_probs >= 0.5).astype(int)))

        # 3. Test Metrics
        test_preds = model.predict({"clinical_embedding": X_clin_test, "image_embedding": X_img_test}, verbose=0)
        test_probs = test_preds[0].ravel()
        test_auc = float(roc_auc_score(y_p_test, test_probs))
        test_acc = float(accuracy_score(y_p_test, (test_probs >= 0.5).astype(int)))

        # Detailed Classification Report for Test
        test_cls_report = classification_report(y_p_test, (test_probs >= 0.5).astype(int), output_dict=True)

        print(f" -> TRAIN AUC: {train_auc:.4f} | TRAIN ACCURACY: {train_acc:.4f}")
        print(f" -> VAL AUC:   {val_auc:.4f} | VAL ACCURACY:   {val_acc:.4f}")
        print(f"🌟 TEST AUC:   {test_auc:.4f} | TEST ACCURACY:  {test_acc:.4f}")

        # Save Final Assets
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "fusion_model.keras"))
        self.save_safetensors(model, os.path.join(output_dir, "fusion_model.safetensors"))

        # Save feature map for deployment (Crucial for the Report Generator)
        with open(os.path.join(output_dir, "secondary_features_map.json"), "w") as f:
            json.dump(self.secondary_features, f)

        # Save Comprehensive Performance Report
        report = {
            "model_type": "Multi-Task Fusion Network",
            "train_metrics": {
                "auc": train_auc,
                "accuracy": train_acc
            },
            "val_metrics": {
                "auc": val_auc,
                "accuracy": val_acc
            },
            "test_metrics": {
                "auc": test_auc,
                "accuracy": test_acc,
                "classification_report": test_cls_report
            },
            "secondary_targets_predicted": self.secondary_features
        }
        with open(os.path.join(output_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=4)

        print(f"Phase 3 Assets saved to {output_dir}")
        return test_auc