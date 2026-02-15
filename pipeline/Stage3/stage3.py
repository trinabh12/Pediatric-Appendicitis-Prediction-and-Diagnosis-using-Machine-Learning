import os
from data_preparation import DataPreparation
from image_processor import ImageProcessor

PREV_STAGE = "../Stage2"
DATASET_DIR = "validation_and_profiling_data"

FILE_NAME = "validated_data.xlsx"
DATA_INFO = "validated_feature_info.json"
MISSING_REPORT = "missing_data.json"
GROUPED_REPORT = "validated_feature_grouped.json"

data_preparation = DataPreparation(PREV_STAGE, DATASET_DIR, FILE_NAME, DATA_INFO, MISSING_REPORT, GROUPED_REPORT)


ORIGINAL_DATASET = "../Regensburg Pediatric Appendicitis Dataset"
IMAGE_DIR = "US_Pictures"
MULTIPLE_IN_ONE = "multiple_in_one"

img_preparation = ImageProcessor(ORIGINAL_DATASET, MULTIPLE_IN_ONE, IMAGE_DIR)

OUTPUT_DIR = "prepared dataset"
os.makedirs(OUTPUT_DIR)

output_tab_dir = os.path.join(OUTPUT_DIR, "tabular")
output_img_dir = os.path.join(OUTPUT_DIR, "image")

print(data_preparation.preparation(output_tab_dir))
print(img_preparation.process_image_data(output_img_dir))
