from feature_handling import HandleFeatures

PREV_STAGE = "../Stage3"
DATASET_DIR = "prepared_dataset"

FILE_NAME = "prepared_data.xlsx"


data_preparation = DataPreparation(PREV_STAGE, DATASET_DIR, FILE_NAME, DATA_INFO, MISSING_REPORT, GROUPED_REPORT)
print(data_preparation.preparation("prepared dataset"))
