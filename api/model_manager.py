import os
import shutil


def copy_pipeline_models(pipeline_dir, local_dir):

    print(f"Starting file transfer from {pipeline_dir} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)

    # Define the exact source paths based on Stage 6 output structure
    phase1_dir = os.path.join(pipeline_dir, "phase1-results", "part_2")
    phase2_dir = os.path.join(pipeline_dir, "phase2-results")
    phase3_dir = os.path.join(pipeline_dir, "Final_Output")

    # WE CHANGED THIS: Now grabbing the native .keras files!
    artifact_mapping = [
        (os.path.join(phase1_dir, "model.keras"), "clinical_model.keras"),
        (os.path.join(phase1_dir, "scaler.pkl"), "scaler.pkl"),
        (os.path.join(phase1_dir, "feature_selector.pkl"), "feature_selector.pkl"),
        (os.path.join(phase2_dir, "model.keras"), "image_model.keras"),
        (os.path.join(phase3_dir, "fusion_model.keras"), "fusion_model.keras"),
        (os.path.join(phase3_dir, "secondary_features_map.json"), "secondary_features_map.json"),
    ]

    for src_path, filename in artifact_mapping:
        dst_path = os.path.join(local_dir, filename)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find {src_path} in pipeline!")

        print(f"   -> Copying {filename}...")
        shutil.copy2(src_path, dst_path)

    print("Successfully copied all models to the local API folder.")