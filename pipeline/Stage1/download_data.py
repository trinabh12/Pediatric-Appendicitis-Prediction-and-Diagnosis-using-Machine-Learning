import os
import requests
import zipfile


class DataDownloader:
    def __init__(self, target_folder):
        self.target_folder = target_folder
        self.excel_url = "https://zenodo.org/records/7669442/files/app_data.xlsx"
        self.zip_url = "https://zenodo.org/records/7669442/files/US_Pictures.zip"

    def download_all(self):
        """Sole purpose: Download Excel and Images zip."""
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        # 1. Download Excel
        excel_path = os.path.join(self.target_folder, "app_data.xlsx")
        self._request_download(self.excel_url, excel_path)

        # 2. Download and Extract Images
        zip_path = os.path.join(self.target_folder, "temp_images.zip")
        self._request_download(self.zip_url, zip_path)

        print(f"[EXTRACTING] {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.target_folder)

        os.remove(zip_path)  # Clean up zip after extraction
        return "[SUCCESS] All data downloaded and extracted."

    def _request_download(self, url, save_path):

        """Fixed the PermissionError by ensuring save_path is a file, not a folder."""
        print(f"[DOWNLOADING] {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {url}. Status: {response.status_code}")

