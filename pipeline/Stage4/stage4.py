import os
import sys
from feature_handling import HandleFeatures

# Path Configurations
PREV_STAGE = "../Stage3"
DATASET_DIR = "prepared dataset"

# Input paths (from Stage 3 output)
TABULAR_INPUT = os.path.join(DATASET_DIR, "tabular")
FILE_NAME = "prepared_data.xlsx"
INFO_REPORT = "feature_info.json"
DERIVED_INFO_REPORT = "derived_info.json"
GROUPED_REPORT = "feature_groups.json"

IMAGE_INPUT = os.path.join(DATASET_DIR, "image")
IMAGE_REGISTRY = os.path.join(PREV_STAGE, IMAGE_INPUT, "image_registry.json")

# Output paths for Stage 4
OUTPUT_DIR = "engineered features"
OUTPUT_TABULAR = os.path.join(OUTPUT_DIR, "tabular")
# Note: We pass the root OUTPUT_DIR to transfer_image_data to manage its own 'image' subfolder
OUTPUT_IMAGE_ROOT = OUTPUT_DIR


def run_stage4():
    print(f"[STAGE 4] Starting Feature Engineering and Multimodal Merging...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Feature Handler
    fe = HandleFeatures(
        PREV_STAGE,
        TABULAR_INPUT,
        FILE_NAME,
        INFO_REPORT,
        DERIVED_INFO_REPORT,
        GROUPED_REPORT,
        IMAGE_INPUT,
        IMAGE_REGISTRY
    )

    # 1. Execute Clinical Feature Engineering & Image Path Merging
    print(f"[STAGE 4] Applying medical thresholds and merging image paths...")
    tab_status = fe.save_data(OUTPUT_TABULAR)

    if tab_status == 0:
        print(f"[STAGE 4] Tabular engineering and Master Manifest saved to '{OUTPUT_TABULAR}'.")
    else:
        print(f"[STAGE 4] Error during tabular engineering.")
        sys.exit(1)

    # 2. Physically transfer image assets (Excluding JSON)
    print(f"[STAGE 4] Transferring View-specific image assets...")
    img_status = fe.transfer_image_data(OUTPUT_IMAGE_ROOT)

    if img_status == 0:
        print(f"[STAGE 4] Image asset transfer successful.")
    else:
        print(f"[STAGE 4] Warning: Image transfer encountered issues.")

    print(f"[STAGE 4] Feature Engineering Stage complete. Data ready for Training.")


if __name__ == "__main__":
    run_stage4()
