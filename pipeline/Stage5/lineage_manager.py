import os
import json
import shutil
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split


class LineageManager:
    def __init__(self, prev_stage, data_dir, file_name, encoding_report, feature_report, img_dir):
        self.tab_dir = os.path.join(prev_stage, data_dir)
        self.img_dir = os.path.join(prev_stage, img_dir)

        self.xlsx_path = os.path.join(self.tab_dir, file_name)
        with open(os.path.join(self.tab_dir, encoding_report), 'r') as file:
            self.encoding_report = json.load(file)
        with open(os.path.join(self.tab_dir, feature_report), 'r') as file:
            self.feature_report = json.load(file)

        if not os.path.exists(self.xlsx_path):
            raise FileNotFoundError(f"Manifest not found: {self.xlsx_path}")
        self.df = pd.read_excel(self.xlsx_path)

    def generate_content_hash(self):
        content_string = self.df.to_string().encode()
        return hashlib.sha256(content_string).hexdigest()[:16]  # 16-char version ID

    def run_versioning_and_split(self, output_folder):
        out_tab = os.path.join(output_folder, "tabular")
        out_img = os.path.join(output_folder, "image")
        os.makedirs(out_tab, exist_ok=True)

        print(f"Task 1: Creating versioned tabular splits...")

        # 2. Stratified Data Partitioning (80/10/10 split)
        # First split: 80% for Training/Validation, 20% for Final Testing
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=0.20,
            stratify=self.df['Diagnosis'],
            random_state=42
        )

        # Second split: From that 80%, take 12.5% to get 10% of the total for Validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.125,
            stratify=train_val_df['Diagnosis'],
            random_state=42
        )

        # 3. CSV Serialization
        # We save as CSV for faster loading in the DataLoader
        train_df.to_csv(os.path.join(out_tab, "train_split.csv"), index=False, encoding='utf-8')
        val_df.to_csv(os.path.join(out_tab, "val_split.csv"), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(out_tab, "test_split.csv"), index=False, encoding='utf-8')

        # 4. Metadata Persistence
        # Saving the rules needed to interpret the features
        with open(os.path.join(out_tab, "encoding_map.json"), 'w', encoding='utf-8') as f:
            json.dump(self.encoding_report, f, indent=4)
        with open(os.path.join(out_tab, "engineered_feature_groups.json"), 'w', encoding='utf-8') as f:
            json.dump(self.feature_report, f, indent=4)

        print(f"Task 2: Transferring image assets to {out_img}...")
        if os.path.exists(self.img_dir):
            for item in os.listdir(self.img_dir):
                item = str(item)
                s_item_path = os.path.join(self.img_dir, item)
                # Only copy 'View_X' folders to maintain structure
                if os.path.isdir(s_item_path) and item.startswith("View_"):
                    d_item_path = os.path.join(out_img, item)
                    if os.path.exists(d_item_path):
                        shutil.rmtree(d_item_path)
                    shutil.copytree(s_item_path, d_item_path)
            print(f"Step 2: Success. Transferred image assets to {out_img}")
        else:
            print(f"[WARNING] Image directory not found at {self.img_dir}")

        # 5. Lineage Fingerprinting
        version_id = self.generate_content_hash()
        lineage_report = {
            "version_id": version_id,
            "split_statistics": {
                "train_count": len(train_df),
                "val_count": len(val_df),
                "test_count": len(test_df)
            },
            "class_distribution": self.df['Diagnosis'].value_counts().to_dict(),
            "source_tabular": self.tab_dir,
            "source_image": self.img_dir
        }

        with open(os.path.join(out_tab, "lineage_report.json"), "w", encoding='utf-8') as f:
            json.dump(lineage_report, f, indent=4)

        print(f"Task 1: Success. Version {version_id} locked. Splits saved to {out_tab}")
        return version_id
