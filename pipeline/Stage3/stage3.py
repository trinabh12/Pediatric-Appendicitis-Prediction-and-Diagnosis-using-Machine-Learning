from data_preparation import DataPreparation

PREV_STAGE = "../Stage2"
DATASET_DIR = "validation_and_profiling_data"

FILE_NAME = "validated_data.xlsx"
DATA_INFO = "validated_feature_info.json"
MISSING_REPORT = "missing_data.json"





dp = DataPreparation(PREV_STAGE, DATASET_DIR, FILE_NAME, DATA_INFO, MISSING_REPORT)
dp.drop_empty_rows()
dp.find_all_dependencies()
