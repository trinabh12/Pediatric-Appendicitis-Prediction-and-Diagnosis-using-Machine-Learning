from feature_handling import HandleFeatures

PREV_STAGE = "../Stage3"
DATASET_DIR = "prepared_dataset"

FILE_NAME = "prepared_data.xlsx"
INFO_REPORT = "feature_info.json"
DERIVED_INFO_REPORT = "derived_info.json"
GROUPED_REPORT = "feature_groups.json"

fe = HandleFeatures(PREV_STAGE, DATASET_DIR, FILE_NAME, INFO_REPORT,DERIVED_INFO_REPORT, GROUPED_REPORT)
