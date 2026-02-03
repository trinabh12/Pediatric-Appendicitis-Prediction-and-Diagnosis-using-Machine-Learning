import os
import json
import pandas as pd
import numpy as np


class DataPreparation:
    def __init__(self, prev_stage, data_dir, data_file, info_report, missing_report):
        self.prev_stage_dir = os.path.join(prev_stage, data_dir)
        self.df = pd.read_excel(os.path.join(self.prev_stage_dir, data_file))
        with open(os.path.join(self.prev_stage_dir, info_report), 'r') as f:
            self.feature_info = json.load(f)
        with open(os.path.join(self.prev_stage_dir, missing_report), 'r') as f:
            self.missing_counts = json.load(f)

        self.dependents = ["BMI", "Alvarado_Score", "Paedriatic_Appendicitis_Score"]

    def drop_empty_rows(self):
        demo_cols = ["Age", "Sex", "Weight", "Height"]

        rows_to_drop = []

        for idx in range(len(self.df)):
            row = self.df.loc[idx, demo_cols]

            if row.isna().all():
                rows_to_drop.append(idx)

        df_cleaned = self.df.drop(index=rows_to_drop).reset_index(drop=True)
        self.df = df_cleaned
        self.missing_counts = {key: value - 1 for key, value in self.missing_counts.items()}

        return self.missing_counts

    def handle_numeric_features(self):
        all_numerics = [feature for feature, f_type in self.feature_info.items()
                        if f_type == "Discrete" or f_type == "Continuous"]

        numeric_features = [feature for feature in all_numerics if feature not in self.dependents]

        low_missing = []
        moderate_missing = []
        high_missing = []

        for feature in numeric_features:
            percent_missing = (self.missing_counts[feature]/len(self.df))*100

            if percent_missing < 5:
                low_missing.append(feature)

            elif percent_missing <= 40:
                moderate_missing.append(feature)

            else:
                high_missing.append(feature)

        print(f"for low missing (Median): {low_missing}")
        print(f"for moderate missing (Iterative): {moderate_missing}")
        print(f"for high missing (Constant/Flag): {high_missing}")


        return numeric_features

    def handle_freetext_features(self):
        freetext_features = [feature for feature, f_type in self.feature_info.items()
                             if f_type == "FreeText"]
        print(len(freetext_features))
        return freetext_features

    def handle_binary_features(self):
        binary_features = [feature for feature, f_type in self.feature_info.items()
                           if (isinstance(f_type, dict) and "Binary" in f_type)]

        print(len(binary_features))
        return binary_features

    def handle_categorical_features(self):
        categorical_features = [feature for feature, f_type in self.feature_info.items()
                                if (isinstance(f_type, dict) and "Categorical" in f_type)]

        print(len(categorical_features))
        return categorical_features

    def handle_image_features(self):
        image_features = [feature for feature, f_type in self.feature_info.items()
                          if f_type == "Image BMP"]

        print(len(image_features))
        return image_features







