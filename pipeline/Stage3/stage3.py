from data_preparation import DataPreparation

PREV_STAGE = "../Stage2"
DATASET_DIR = "validation_and_profiling_data"

FILE_NAME = "validated_data.xlsx"
DATA_INFO = "validated_feature_info.json"
MISSING_REPORT = "missing_data.json"
GROUPED_REPORT = "validated_feature_grouped.json"

dp = DataPreparation(PREV_STAGE, DATASET_DIR, FILE_NAME, DATA_INFO, MISSING_REPORT, GROUPED_REPORT)

print(dp.handle_demographic())
print(dp.handle_clinical())

