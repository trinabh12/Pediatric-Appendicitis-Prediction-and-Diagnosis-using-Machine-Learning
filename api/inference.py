import os
import cv2
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from model_manager import copy_pipeline_models


# --- Inference Engine ---
class AppendicitisPredictor:
    def __init__(self, pipeline_dir, local_dir):
        self.local_dir = local_dir

        # 1. CHECK IF FILES EXIST (Looking for .keras now)
        expected_files = [
            "clinical_model.keras", "scaler.pkl", "feature_selector.pkl",
            "image_model.keras", "fusion_model.keras", "secondary_features_map.json"
        ]

        missing_files = False
        if not os.path.exists(self.local_dir):
            missing_files = True
        else:
            for f in expected_files:
                if not os.path.exists(os.path.join(self.local_dir, f)):
                    missing_files = True
                    break

        # 2. TRIGGER COPY IF NECESSARY
        if missing_files:
            print(f"⚠️ Local models missing in '{self.local_dir}'. Initiating copy sequence...")
            copy_pipeline_models(pipeline_dir, self.local_dir)
            print("✅ Resuming engine initialization...")
        else:
            print(f"✅ All model files found locally in '{self.local_dir}'.")

        # 3. LOAD ARTIFACTS INTO RAM (Natively with Keras)
        print("🧠 Loading AI Models into memory...")
        self.scaler = joblib.load(os.path.join(self.local_dir, "scaler.pkl"))
        self.selector = joblib.load(os.path.join(self.local_dir, "feature_selector.pkl"))

        with open(os.path.join(self.local_dir, "secondary_features_map.json"), "r") as f:
            self.secondary_features = json.load(f)

        # Load the models directly! No architecture building required.
        self.clinical_model = load_model(os.path.join(self.local_dir, "clinical_model.keras"))
        self.image_model = load_model(os.path.join(self.local_dir, "image_model.keras"))
        self.fusion_model = load_model(os.path.join(self.local_dir, "fusion_model.keras"))

        # Extract sub-models for the embeddings
        self.clin_emb_extractor = Model(inputs=self.clinical_model.input,
                                        outputs=self.clinical_model.get_layer("clinical_embedding").output)
        self.img_emb_extractor = Model(inputs=self.image_model.input,
                                       outputs=self.image_model.get_layer("image_embedding").output)

        print("🚀 Engine is HOT and ready for inference!")

    def _process_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file provided.")
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
            resized = cv2.resize(padded, (224, 224))
            rgb = cv2.merge([resized, resized, resized])
            processed.append(rgb.astype(np.float32) / 255.0)

        return np.array(processed)

    def predict(self, tabular_dict, image_bytes):
        raw_features = np.array([list(tabular_dict.values())])
        scaled_features = self.scaler.transform(raw_features)
        selected_features = self.selector.transform(scaled_features)

        clin_emb = self.clin_emb_extractor.predict(selected_features, verbose=0)
        img_array = self._process_image(image_bytes)
        img_embs = self.img_emb_extractor.predict(img_array, verbose=0)
        final_img_emb = np.mean(img_embs, axis=0, keepdims=True)

        fusion_preds = self.fusion_model.predict(
            {"clinical_embedding": clin_emb, "image_embedding": final_img_emb}, verbose=0
        )

        primary_prob = float(fusion_preds[0][0][0])
        report = {
            "Appendicitis_Probability": round(primary_prob * 100, 2),
            "Risk_Level": "HIGH" if primary_prob > 0.614 else "LOW",
            "Secondary_Findings": {}
        }
        for name, prob in zip(self.secondary_features, fusion_preds[1][0]):
            report["Secondary_Findings"][name] = round(float(prob) * 100, 2)

        return report