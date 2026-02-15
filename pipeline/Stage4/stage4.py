import os
from feature_handling import HandleFeatures

PREV_STAGE = "../Stage3"
DATASET_DIR = "prepared dataset"

TABULAR_DATA = os.path.join(DATASET_DIR, "tabular")
FILE_NAME = "prepared_data.xlsx"
INFO_REPORT = "feature_info.json"
DERIVED_INFO_REPORT = "derived_info.json"
GROUPED_REPORT = "feature_groups.json"

IMAGE_DATA = os.path.join(DATASET_DIR, "image")

fe = HandleFeatures(PREV_STAGE, TABULAR_DATA, FILE_NAME, INFO_REPORT, DERIVED_INFO_REPORT, GROUPED_REPORT)
fe.save_data("engineered features")
