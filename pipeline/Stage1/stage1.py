import os
from ingestion import IngestionStage
from download_data import DataDownloader


ROOT_DIR = "../"
os.mkdirs(os.path.join(ROOT_DIR, "data"), exist_ok =True)

DATASET_DIR = "Regensburg Pediatric Appendicitis Dataset"
FILE_NAME = "app_data.xlsx"
IMAGE_DIR = "US_Pictures"
TARGET_FOLDER = os.path.join(ROOT_DIR, "data", "ingestion data")

OUTPUT_TABULAR = os.path.join(TARGET_FOLDER, "tabular")
OUTPUT_IMAGE = os.path.join(TARGET_FOLDER, "image")

raw_data_path = os.path.join(ROOT_DIR, DATASET_DIR, FILE_NAME)


def run_stage1():

    if not os.path.exists(raw_data_path):
        print(f"[STAGE 1] Data not found at {raw_data_path}. Initializing download...")

        downloader = DataDownloader(os.path.join(ROOT_DIR, DATASET_DIR))
        downloader.download_all()
        print("[STAGE 1] Download complete.")
    else:
        print(f"[STAGE 1] Raw data found at {raw_data_path}. Skipping download.")

    print(f"[STAGE 1] Starting ingestion to '{TARGET_FOLDER}'...")
    ingestion = IngestionStage(ROOT_DIR, DATASET_DIR, FILE_NAME, IMAGE_DIR)
    ingestion.extract_tabular(OUTPUT_TABULAR)
    ingestion.extract_image(OUTPUT_IMAGE)

    print("[STAGE 1] Ingestion successful.")


if __name__ == "__main__":
    run_stage1()
