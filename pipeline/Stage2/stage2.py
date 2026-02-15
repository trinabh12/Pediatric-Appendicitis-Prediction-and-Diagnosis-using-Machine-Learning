import os
from validate_data import ValidationAndProfiling

# Path Configurations
PREV_STAGE = "../Stage1"
DATASET_DIR = "ingestion_data"

# Input paths (from Stage 1 output)
TABULAR_INPUT = os.path.join(DATASET_DIR, "tabular")
FILE_NAME = "raw_data.xlsx"
DATA_INFO = "feature_info.json"
DATA_SUMMARY = "feature_summary.json"
DATA_GROUPED = "feature_grouped.json"
IMAGE_INPUT = os.path.join(DATASET_DIR, "image")

# Output paths for Stage 2
TARGET_FOLDER = "validation_and_profiling_data"
OUTPUT_TABULAR = os.path.join(TARGET_FOLDER, "tabular")
OUTPUT_IMAGE = os.path.join(TARGET_FOLDER, "image")


def run_stage2():
    print(f"[STAGE 2] Initializing Validation and Profiling...")

    # Initialize the Validation logic
    validation_check = ValidationAndProfiling(
        PREV_STAGE,
        TABULAR_INPUT,
        FILE_NAME,
        DATA_INFO,
        DATA_SUMMARY,
        DATA_GROUPED,
        IMAGE_INPUT
    )

    # Execute Validation Suite
    print(f"[STAGE 2] Running feature metadata validation...")
    validation_result = validation_check.validation()

    if validation_result is True:
        print("[STAGE 2] Metadata validation successful. Mapping features...")

        validation_check.save_tab_report(OUTPUT_TABULAR)
        print(f"[STAGE 2] Tabular validation reports saved to '{OUTPUT_TABULAR}'.")

        print(f"[STAGE 2] Validating image naming conventions and moving assets...")
        img_status = validation_check.save_img_report(OUTPUT_IMAGE)

        if img_status == 0:
            print(f"[STAGE 2] Image validation and cleanup successful.")
        else:
            print(f"[STAGE 2] Warning: Image report encountered issues.")

        print("[STAGE 2] Validation and Profiling Stage complete.")
    else:
        print(f"[STAGE 2] CRITICAL ERROR: {validation_result}")


if __name__ == "__main__":
    run_stage2()
