import os
import sys
import shutil
from lineage_manager import LineageManager

# Path Configurations
PREV_STAGE = "../Stage4"
DATASET_DIR = "engineered features"

TABULAR_SUB = os.path.join(DATASET_DIR, "tabular")
IMAGE_SUB = os.path.join(DATASET_DIR, "image")

# Input Filenames
XLSX_INPUT = "engineered_data.xlsx"
ENCODING_REPORT = "encoding_map.json"
FEATURE_REPORT = "engineered_feature_groups.json"

# Output Configuration
TARGET_FOLDER = "training dataset"


def run_stage5():
    print(f"[STAGE 5] Initializing Data Lineage and Stratified Splitting...")

    # Ensure clean output directory for the final dataset
    if os.path.exists(TARGET_FOLDER):
        print(f"[STAGE 5] Refreshing target folder: {TARGET_FOLDER}")
        shutil.rmtree(TARGET_FOLDER)
    os.makedirs(TARGET_FOLDER)

    # Initialize the Lineage Manager
    # This handles UTF-8 encoding and stratified splitting logic
    try:
        vl = LineageManager(
            PREV_STAGE,
            TABULAR_SUB,
            XLSX_INPUT,
            ENCODING_REPORT,
            FEATURE_REPORT,
            IMAGE_SUB
        )
    except FileNotFoundError as e:
        print(f"[STAGE 5] CRITICAL ERROR: {e}")
        sys.exit(1)

    version_id = vl.run_versioning_and_split(TARGET_FOLDER)

    print("-" * 30)
    print(f"[STAGE 5] VERSION LOCKED: {version_id}")
    print(f"[STAGE 5] Final dataset prepared in '{TARGET_FOLDER}'")


    print(f"[STAGE 5] Pipeline ready for Stage 6 (Model Training).")


if __name__ == "__main__":
    run_stage5()
