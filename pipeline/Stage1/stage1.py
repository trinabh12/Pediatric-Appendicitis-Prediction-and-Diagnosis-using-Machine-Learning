from ingestion import IngestionStage

ROOT_DIR = "../"
DATASET_DIR = "Regensburg Pediatric Appendicitis Dataset"
FILE_NAME = "app_data.xlsx"

target_folder = "ingestion_data"

ingestion = IngestionStage(ROOT_DIR, DATASET_DIR, FILE_NAME)
ingestion.extract_xlsx(target_folder)
