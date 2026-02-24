import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from safetensors.tensorflow import save_file


def build_image_cnn(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    embedding_layer = Dense(16, activation="relu",
                            kernel_regularizer=regularizers.l2(0.001),
                            name="image_embedding")(x)
    x_emb = BatchNormalization()(embedding_layer)

    outputs = Dense(1, activation="sigmoid", name="diagnosis_output")(x_emb)

    model = Model(inputs=inputs, outputs=outputs, name="Phase2_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


class Phase2:
    def __init__(self, prev_stage, data_dir, train_file, val_file, test_file, seed=0):
        self.base_path = os.path.join(prev_stage, data_dir)
        self.tabular_path = os.path.join(self.base_path, "tabular")
        self.image_path = os.path.join(self.base_path, "image")

        self.train_df = pd.read_csv(os.path.join(self.tabular_path, train_file))
        self.val_df = pd.read_csv(os.path.join(self.tabular_path, val_file))
        self.test_df = pd.read_csv(os.path.join(self.tabular_path, test_file))

        self.view_cols = [c for c in self.train_df.columns if "View" in c and "Path" in c]
        self.target_size = (224, 224)

        tf.random.set_seed(seed)
        np.random.seed(seed)

    def save_safetensors(self, model, output_path):
        weights = model.get_weights()
        weight_names = [v.name for v in model.weights]
        tensors = {name: tf.convert_to_tensor(weight) for name, weight in zip(weight_names, weights)}
        save_file(tensors, output_path)

    def _extract_instances(self, df):
        paths, labels = [], []
        for _, row in df.iterrows():
            if row.get('Has_Images', 0) == 1:
                for col in self.view_cols:
                    p = row[col]
                    if pd.notna(p) and p != "MISSING_IMAGE":
                        p = p.replace('\\', os.sep).replace('/', os.sep)
                        paths.append(os.path.join(self.image_path, p))
                        labels.append(row['Diagnosis'])
        return paths, labels

    def _smart_load_and_split(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []

        h, w = img.shape
        sub_images = []

        if w > 1.5 * h:
            sub_images.append(img[:, :w // 2])
            sub_images.append(img[:, w // 2:])
        else:
            sub_images.append(img)

        processed = []
        for sub in sub_images:
            sh, sw = sub.shape
            max_dim = max(sh, sw)

            top = (max_dim - sh) // 2
            bottom = max_dim - sh - top
            left = (max_dim - sw) // 2
            right = max_dim - sw - left

            padded = cv2.copyMakeBorder(sub, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            resized = cv2.resize(padded, self.target_size)
            rgb = cv2.merge([resized, resized, resized])
            processed.append(rgb.astype(np.float32) / 255.0)

        return processed

    def get_dataset(self, df, batch_size=32, shuffle=False):
        paths, labels = self._extract_instances(df)

        def generator():
            for p, l in zip(paths, labels):
                imgs = self._smart_load_and_split(p)
                for img in imgs:
                    yield img, l
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=500)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def generate_patient_embeddings(self, model, df):
        emb_model = Model(inputs=model.input, outputs=model.get_layer("image_embedding").output)
        patient_embeddings = []

        print(f"Extracting image embeddings for {len(df)} patients...")
        for _, row in df.iterrows():
            valid_paths = []
            if row.get('Has_Images', 0) == 1:
                for col in self.view_cols:
                    p = row[col]
                    if pd.notna(p) and p != "MISSING_IMAGE":
                        p = p.replace('\\', os.sep).replace('/', os.sep)
                        valid_paths.append(os.path.join(self.image_path, p))

            if not valid_paths:
                patient_embeddings.append(np.zeros(16, dtype=np.float32))
            else:
                all_sub_images = []
                for p in valid_paths:
                    all_sub_images.extend(self._smart_load_and_split(p))

                if not all_sub_images:
                    patient_embeddings.append(np.zeros(16, dtype=np.float32))
                else:
                    imgs = np.array(all_sub_images)
                    embs = emb_model.predict(imgs, verbose=0)
                    patient_emb = np.mean(embs, axis=0)  # Average all views and splits
                    patient_embeddings.append(patient_emb)

        return np.array(patient_embeddings)

    def train_phase2(self, output_dir):
        print("Preparing Image Datasets with Smart Split & Pad...")
        train_ds = self.get_dataset(self.train_df, batch_size=32, shuffle=True)
        val_ds = self.get_dataset(self.val_df, batch_size=32, shuffle=False)
        test_ds = self.get_dataset(self.test_df, batch_size=32, shuffle=False)

        model = build_image_cnn()

        early_stop = EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, mode="max")

        print("--- Training Image CNN (Phase 2) ---")
        model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop, lr_scheduler])

        os.makedirs(output_dir, exist_ok=True)

        print("\n--- Evaluating Phase 2 Performance ---")

        eval_train_ds = self.get_dataset(self.train_df, batch_size=32, shuffle=False)

        train_loss, train_acc, train_auc = model.evaluate(eval_train_ds, verbose=0)
        val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)

        print(f"TRAIN -> Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"VAL   -> Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")
        print(f"TEST  -> Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")

        metrics = {
            "train": {
                "accuracy": float(train_acc),
                "auc": float(train_auc),
                "loss": float(train_loss)
            },
            "validation": {
                "accuracy": float(val_acc),
                "auc": float(val_auc),
                "loss": float(val_loss)
            },
            "test": {
                "accuracy": float(test_acc),
                "auc": float(test_auc),
                "loss": float(test_loss)
            }
        }

        with open(os.path.join(output_dir, "phase2_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        model.save(os.path.join(output_dir, "model.keras"))
        self.save_safetensors(model, os.path.join(output_dir, "model.safetensors"))

        print("\nGenerating Patient Embeddings...")
        np.save(os.path.join(output_dir, "train_image_embeddings.npy"),
                self.generate_patient_embeddings(model, self.train_df))
        np.save(os.path.join(output_dir, "val_image_embeddings.npy"),
                self.generate_patient_embeddings(model, self.val_df))
        np.save(os.path.join(output_dir, "test_image_embeddings.npy"),
                self.generate_patient_embeddings(model, self.test_df))

        print(f"\nPhase 2 Assets saved successfully to: {output_dir}")