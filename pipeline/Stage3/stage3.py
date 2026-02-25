import os
import shutil
from data_preparation import DataPreparation
from image_processor import ImageProcessor

# Path Configurations
PREV_STAGE = "../Stage2"
DATASET_DIR = "validation and profiling data"

# Input paths (from Stage 2 output)
TABULAR_INPUT = os.path.join(DATASET_DIR, "tabular")
FILE_NAME = "validated_data.xlsx"
DATA_INFO = "validated_feature_info.json"
MISSING_REPORT = "missing_data.json"
GROUPED_REPORT = "validated_feature_grouped.json"

IMAGE_INPUT = os.path.join(DATASET_DIR, "image")
IMAGE_REPORT = "image_validation_report.json"

# Output paths for Stage 3
OUTPUT_DIR = "prepared dataset"
OUTPUT_TAB_DIR = os.path.join(OUTPUT_DIR, "tabular")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "image")


def run_stage3():
    print(f"[STAGE 3] Starting Data Preparation...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Preparation Objects
    data_prep = DataPreparation(
        PREV_STAGE,
        TABULAR_INPUT,
        FILE_NAME,
        DATA_INFO,
        MISSING_REPORT,
        GROUPED_REPORT
    )

    img_prep = ImageProcessor(
        PREV_STAGE,
        IMAGE_INPUT,
        IMAGE_REPORT
    )

    # 1. Process Tabular Data
    print(f"[STAGE 3] Transforming tabular data and handling missing values...")
    tab_status = data_prep.preparation(OUTPUT_TAB_DIR)
    print(f"Tabular Preparation Status: {tab_status}")

    # 2. Process Image Data (Any to BMP, Collision Handling, Segregation)
    print(f"[STAGE 3] Standardizing images and segregating views...")
    img_status = img_prep.process_image_data(OUTPUT_IMG_DIR)

    if img_status:
        print(f"[STAGE 3] Image preparation successful. Registry created.")
    else:
        print(f"[STAGE 3] Error encountered during image preparation.")

    # 3. Cleanup: Remove residual directory
    # We check if it exists and is empty (or contains only processed sub-folders)
    residual_path = DATASET_DIR
    if os.path.exists(residual_path):
        print(f"[STAGE 3] Cleaning up residual folder: {residual_path}")
        try:
            shutil.rmtree(residual_path)
            print(f"[STAGE 3] Residual data removed successfully.")
        except Exception as e:
            print(f"[STAGE 3] Warning: Could not remove residual folder: {e}")

    print("[STAGE 3] Preparation Stage complete.")


if __name__ == "__main__":
    run_stage3()
