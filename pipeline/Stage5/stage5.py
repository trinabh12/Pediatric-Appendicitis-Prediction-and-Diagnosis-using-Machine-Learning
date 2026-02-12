import os
import pandas as pd
import json
import hashlib
from datetime import datetime
import shutil


class DataVersioning:
    def __init__(self, data_path, metadata_dir, version_root="data_versions"):
        self.data_path = data_path
        self.metadata_dir = metadata_dir
        self.version_root = version_root

        # Load the data to generate stats
        self.df = pd.read_excel(data_path)

        # Generate a unique Version Tag based on timestamp
        self.version_tag = datetime.now().strftime("v_%Y%m%d_%H%M%S")
        self.version_dir = os.path.join(self.version_root, self.version_tag)

    def _calculate_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_lineage_report(self, stage_name="Feature Engineering"):
        report = {
            "version_tag": self.version_tag,
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "data_fingerprint": self._calculate_hash(self.data_path),
            "statistics": {
                "row_count": int(self.df.shape[0]),
                "feature_count": int(self.df.shape[1]),
                "target_balance": self.df['Diagnosis'].value_counts(normalize=True).to_dict()
                if 'Diagnosis' in self.df.columns else "N/A"
            },
            "input_files": [
                os.path.basename(self.data_path),
                "encoding_map.json",
                "feature_groups_final.json"
            ]
        }
        return report

    def finalize_version(self):
        """Archives the data, metadata, and lineage report into a versioned folder."""
        if not os.path.exists(self.version_dir):
            os.makedirs(self.version_dir)

        # 1. Generate and save the lineage report
        report = self.create_lineage_report()
        with open(os.path.join(self.version_dir, "lineage_manifest.json"), "w") as f:
            json.dump(report, f, indent=4)

        # 2. Copy the data file
        shutil.copy(self.data_path, os.path.join(self.version_dir, "engineered_data.xlsx"))

        # 3. Copy the metadata files
        metadata_files = ["encoding_map.json", "feature_groups_final.json"]
        for meta_file in metadata_files:
            src = os.path.join(self.metadata_dir, meta_file)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.version_dir, meta_file))

        # 4. Update the Master Lineage Log (Central registry of all versions)
        master_log_path = os.path.join(self.version_root, "master_lineage_log.json")
        master_log = []

        if os.path.exists(master_log_path):
            with open(master_log_path, "r") as f:
                master_log = json.load(f)

        master_log.append({
            "version": self.version_tag,
            "path": self.version_dir,
            "timestamp": report["timestamp"],
            "hash": report["data_fingerprint"]
        })

        with open(master_log_path, "w") as f:
            json.dump(master_log, f, indent=4)

        print(f"Stage 5 Success: Version {self.version_tag} archived with full lineage tracking.")
        return self.version_tag