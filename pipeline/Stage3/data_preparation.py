import os
import json
import pandas as pd
#import numpy as np


class DataPreparation:
    def __init__(self, prev_stage, data_dir, data_file, info_report, missing_report, grouped_report):
        self.prev_stage_dir = os.path.join(prev_stage, data_dir)
        self.df = pd.read_excel(os.path.join(self.prev_stage_dir, data_file))

        with open(os.path.join(self.prev_stage_dir, info_report), 'r') as f:
            self.feature_info = json.load(f)
        with open(os.path.join(self.prev_stage_dir, missing_report), 'r') as f:
            self.missing_counts = json.load(f)
        with open(os.path.join(self.prev_stage_dir, grouped_report), 'r') as f:
            self.grouped_features = json.load(f)

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

    def feature_value_type(self, feature_name):
        if self.feature_info[feature_name] == "Discrete":
            return "numeric"
        if self.feature_info[feature_name] == "Continuous":
            return "numeric"
        if isinstance(self.feature_info[feature_name], dict):
            if "Binary" in self.feature_info[feature_name]:
                return "bin_or_cat"
            if "Categorical" in self.feature_info[feature_name]:
                return "bin_or_cat"

    def binary_or_categorical_groups(self, feature_name):
        if self.feature_value_type(feature_name) != "bin_or_cat":
            return f"{feature_name} is not a Binary or Categorical feature."

        info = self.feature_info[feature_name]
        if "Binary" in info:
            groups = info["Binary"]
            return groups

        if "Categorical" in info:
            groups = info["Categorical"]
            return groups

    def numeric_filling(self, feature_name):
        if feature_name not in self.df.columns:
            return "feature name not found."

        if self.feature_value_type(feature_name) != "numeric":
            return "feature is not a numeric."

        total_rows = len(self.df)
        missing_pct = (self.missing_counts[feature_name] / total_rows) * 100

        if missing_pct >50:
            self.df[feature_name] = self.df[feature_name].fillna(0)
            return f"{feature_name}: Filled null values with 0"

        if 50 >= missing_pct > 5:
            self.df[feature_name] = self.df[feature_name].fillna(0)
            return f"{feature_name}: Filled null values with 0"

        if missing_pct <= 5:
            skewness = self.df[feature_name].skew()

            if abs(skewness) > 1:
                fill_value = self.df[feature_name].median()
                self.df[feature_name] = self.df[feature_name].fillna(fill_value)
                return f"{feature_name}: Filled with median"
            else:
                fill_value = self.df[feature_name].mean()
                self.df[feature_name] = self.df[feature_name].fillna(fill_value)
                return f"{feature_name}: Filled with mean"
        else:
            return f"{feature_name}: Does not need filling"

    def binary_and_categorical_filling(self, feature_name):
        if self.feature_value_type(feature_name) != "bin_or_cat":
            return f"{feature_name} is not a Binary or Categorical feature."

        total_rows = len(self.df)
        missing_count = self.missing_counts.get(feature_name, 0)

        if missing_count == 0:
            return f"{feature_name}: No missing values."

        missing_pct = (missing_count / total_rows) *100

        if missing_pct > 50:
            self.df[feature_name] = self.df[feature_name].fillna(0)
            return f"{feature_name}: Filled null values with 0"

        fill_value = self.df[feature_name].mode()[0]
        self.df[feature_name] = self.df[feature_name].fillna(fill_value)
        return f"{feature_name}: Filled with Mode"

    def handle_demographic(self):
        dependent_features = ["BMI"]
        key_group = "Demographic / Other"

        feature_list = [feature for feature in self.grouped_features[key_group]
                        if feature not in dependent_features]



        for feature in feature_list:
            if self.feature_value_type(feature) == "numeric":
                log = self.numeric_filling(feature)
                print(log)


            elif self.feature_value_type(feature) == "bin_or_cat":
                log = self.binary_and_categorical_filling(feature)
                print(log)

        if "Height" in self.df.columns and "Weight" in self.df.columns:
            # BMI = weight(kg) / (height(m)^2)
            mask = self.df["BMI"].isna() & self.df["Height"].notna() & self.df["Weight"].notna()
            self.df.loc[mask, "BMI"] = self.df["Weight"] / ((self.df["Height"] / 100) ** 2)
            print("BMI: Filled with h and w")

            if self.df["BMI"].isna().any():
                self.numeric_filling("BMI")
                print("BMI: Recalculated where possible, remaining imputed.")

        return 0
