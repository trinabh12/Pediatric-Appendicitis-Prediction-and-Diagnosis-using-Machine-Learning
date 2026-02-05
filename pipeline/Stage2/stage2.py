from validate_data import ValidationAndProfiling

target_folder = "validation_and_profiling_data"

PREV_STAGE = "../Stage1"
DATASET_DIR = "ingestion_data"
FILE_NAME = "raw_data.xlsx"
DATA_INFO = "feature_info.json"
DATA_SUMMARY = "feature_summary.json"
DATA_GROUPED = "feature_grouped.json"



validation_check = ValidationAndProfiling(PREV_STAGE, DATASET_DIR, FILE_NAME, DATA_INFO, DATA_SUMMARY, DATA_GROUPED)

if validation_check.validation():
    validation_check.save_report(target_folder)

