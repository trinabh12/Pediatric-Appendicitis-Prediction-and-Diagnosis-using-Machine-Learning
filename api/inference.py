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

        # 1. EXPECTED ARTIFACTS
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
            print(f"Local models missing in '{self.local_dir}'. Initiating copy sequence...")
            copy_pipeline_models(pipeline_dir, self.local_dir)
            print("Resuming engine initialization...")
        else:
            print(f"All model files found locally in '{self.local_dir}'.")

        # 3. LOAD ARTIFACTS INTO RAM
        print("Loading AI Models into memory...")
        self.scaler = joblib.load(os.path.join(self.local_dir, "scaler.pkl"))
        self.selector = joblib.load(os.path.join(self.local_dir, "feature_selector.pkl"))

        with open(os.path.join(self.local_dir, "secondary_features_map.json"), "r") as f:
            self.secondary_features = json.load(f)

        # Load the models natively with Keras
        self.clinical_model = load_model(os.path.join(self.local_dir, "clinical_model.keras"))
        self.image_model = load_model(os.path.join(self.local_dir, "image_model.keras"))
        self.fusion_model = load_model(os.path.join(self.local_dir, "fusion_model.keras"))

        # Extract sub-models for the embeddings
        self.clin_emb_extractor = Model(inputs=self.clinical_model.input,
                                        outputs=self.clinical_model.get_layer("clinical_embedding").output)
        self.img_emb_extractor = Model(inputs=self.image_model.input,
                                       outputs=self.image_model.get_layer("image_embedding").output)

        print("Engine is HOT and ready for inference!")

    def _derive_features(self, data):
        """
        Manually re-calculate derived features from Stage 5.
        Uses .get(key, 0.0) to prevent crashes if a field is missing.
        """
        # --- clinical_derived ---
        data["Classic_Presentation_Flag"] = 1.0 if (data.get("Migratory_Pain", 0) == 1 and
                                                    data.get("Nausea", 0) == 1 and
                                                    data.get("Loss_of_Appetite", 0) == 1) else 0.0

        data["Fever_Flag"] = 1.0 if data.get("Body_Temperature", 36.5) > 37.5 else 0.0

        # --- lab_derived ---
        data["Inflammatory_Triage"] = 1.0 if (data.get("WBC_Count", 0) > 10 and
                                              data.get("CRP", 0) > 10) else 0.0

        data["Left_Shift_Signal"] = 1.0 if data.get("Neutrophil_Percentage", 0) > 75 else 0.0

        data["High_CRP_Flag"] = 1.0 if data.get("CRP", 0) > 20 else 0.0

        # --- ultrasound_derived (Placeholders for UI) ---
        data["Has_Images"] = 1.0
        data["US_Sequence_Count"] = 1.0
        # Secondary findings score usually depends on image analysis;
        # setting a neutral default here as the image model handles the heavy lifting
        if "Secondary_Findings_Score" not in data:
            data["Secondary_Findings_Score"] = 0.0

        return data

    def _process_image(self, image_bytes):
        """Preprocesses image exactly as training."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file.")

        h, w = img.shape
        sub_images = [img[:, :w // 2], img[:, w // 2:]] if w > 1.5 * h else [img]

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
        # 1. RUN AUTO-DERIVATION
        full_data = self._derive_features(tabular_dict)

        # 2. ALIGN WITH SCALER (Critical for preventing 100% bug)
        # Using .get(name, 0.0) ensures it won't crash on missing keys
        try:
            ordered_values = [full_data.get(name, 0.0) for name in self.scaler.feature_names_in_]
            raw_features = np.array([ordered_values])
        except Exception as e:
            raise ValueError(f"Feature alignment failed: {e}")

        # 3. TRANSFORM DATA
        scaled_features = self.scaler.transform(raw_features)
        selected_features = self.selector.transform(scaled_features)

        # 4. GENERATE EMBEDDINGS
        clin_emb = self.clin_emb_extractor.predict(selected_features, verbose=0)
        img_array = self._process_image(image_bytes)
        img_embs = self.img_emb_extractor.predict(img_array, verbose=0)
        final_img_emb = np.mean(img_embs, axis=0, keepdims=True)

        # 5. FUSION INFERENCE
        fusion_preds = self.fusion_model.predict(
            {"clinical_embedding": clin_emb, "image_embedding": final_img_emb},
            verbose=0
        )

        # 6. PARSE DIAGNOSIS
        primary_prob = float(np.squeeze(fusion_preds[0]))

        report = {
            "Appendicitis_Probability": round(primary_prob * 100, 2),
            "Risk_Level": "HIGH" if primary_prob > 0.614 else "LOW",
            "Secondary_Findings": {}
        }

        # 7. PARSE SECONDARY FINDINGS
        secondary_probs = np.squeeze(fusion_preds[1])
        for name, prob in zip(self.secondary_features, secondary_probs):
            report["Secondary_Findings"][name] = round(float(prob) * 100, 2)

        return report