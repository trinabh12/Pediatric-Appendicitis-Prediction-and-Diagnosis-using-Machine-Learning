import os
import json
import pandas as pd
import numpy as np


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
        numeric_dependencies = ["BMI", "Alvarado_Score", "Paedriatic_Appendicitis_Score"]
        ultrasound_features = self.grouped_features["Ultrasound"]

        if feature_name in numeric_dependencies:
            return "numerically_dependent"
        if feature_name in ultrasound_features:
            return "ultrasound_feature"
        if self.feature_info[feature_name] == "Discrete":
            return "numeric"
        if self.feature_info[feature_name] == "Continuous":
            return "numeric"
        if isinstance(self.feature_info[feature_name], dict):
            if "Binary" in self.feature_info[feature_name]:
                return "bin_or_cat"
            if "Categorical" in self.feature_info[feature_name]:
                return "bin_or_cat"
        if self.feature_info[feature_name] == "FreeText":
            return "freetext"
        if self.feature_info[feature_name] == "Image BMP":
            return "BMP"

    def numeric_filling(self, feature_name):
        if feature_name not in self.df.columns:
            return "feature name not found."

        if self.feature_value_type(feature_name) != "numeric":
            return "feature is not a numeric."

        total_rows = len(self.df)
        missing_pct = (self.missing_counts[feature_name] / total_rows) * 100

        if missing_pct >50:
            self.df[feature_name] = self.df[feature_name].fillna(0)
            return f"{feature_name}: Filled null values with 0."

        if 50 >= missing_pct > 10:
            fill_value = self.df[feature_name].median()
            self.df[feature_name] = self.df[feature_name].fillna(fill_value)
            return f"{feature_name}: Filled null values with median."

        if missing_pct <= 10:
            skewness = self.df[feature_name].skew()

            if abs(skewness) > 1:
                fill_value = self.df[feature_name].median()
                self.df[feature_name] = self.df[feature_name].fillna(fill_value)
                return f"{feature_name}: Filled with median."
            else:
                fill_value = self.df[feature_name].mean()
                self.df[feature_name] = self.df[feature_name].fillna(fill_value)
                return f"{feature_name}: Filled with mean."
        else:
            return f"{feature_name}: Does not need filling."

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
            return f"{feature_name}: Filled null values with 0."

        fill_value = self.df[feature_name].mode()[0]
        self.df[feature_name] = self.df[feature_name].fillna(fill_value)
        return f"{feature_name}: Filled with Mode."

    def freetext_and_imginfo_filling(self, feature_name):
        if self.feature_value_type(feature_name) != "freetext":
            return f"{feature_name} is not a free text"

        self.df[feature_name] = self.df[feature_name].fillna("none")

        return f"{feature_name}: Filled with missing indicator."


    def handle_features(self, feature_group):
        all_features = self.grouped_features[feature_group]

        for feature in all_features:
            value_type = self.feature_value_type(feature)

            if value_type == "bin_or_cat":
                self.binary_and_categorical_filling(feature)
            elif value_type == "numeric":
                self.numeric_filling(feature)
            elif value_type in ["freetext", "BMP"]:
                self.freetext_and_imginfo_filling(feature)

    def handle_dependent_features(self):
        def score_val(col):
            if col not in self.df.columns: return 0
            return self.df[col].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

        mask_alv = self.df["Alvarado_Score"].isna()
        if mask_alv.any():
            neut_pt = (self.df["Neutrophil_Percentage"].fillna(0) >= 75).astype(int)
            wbc_pt = (self.df["WBC_Count"].fillna(0) > 10).astype(int) * 2
            temp_pt = (self.df["Body_Temperature"].fillna(0) >= 37.3).astype(int)

            self.df.loc[mask_alv, "Alvarado_Score"] = (
                    score_val("Migratory_Pain") + score_val("Loss_of_Appetite") +
                    score_val("Nausea") + (score_val("Lower_Right_Abd_Pain") * 2) +
                    score_val("Ipsilateral_Rebound_Tenderness") +
                    temp_pt + wbc_pt + neut_pt
            )


        mask_pas = self.df["Paedriatic_Appendicitis_Score"].isna()
        if mask_pas.any():
            neut_pt = (self.df["Neutrophil_Percentage"].fillna(0) >= 75).astype(int)
            wbc_pt = (self.df["WBC_Count"].fillna(0) > 10).astype(int) * 2
            temp_pt = (self.df["Body_Temperature"].fillna(0) >= 38.0).astype(int)

            self.df.loc[mask_pas, "Paedriatic_Appendicitis_Score"] = (
                    (score_val("Coughing_Pain") * 2) + (score_val("Lower_Right_Abd_Pain") * 2) +
                    score_val("Nausea") + score_val("Loss_of_Appetite") +
                    temp_pt + wbc_pt + neut_pt + score_val("Migratory_Pain")
            )

        mask_bmi = self.df["BMI"].isna() & self.df["Height"].notna() & self.df["Weight"].notna()
        self.df.loc[mask_bmi, "BMI"] = self.df["Weight"] / ((self.df["Height"] / 100) ** 2)
        if self.df["BMI"].isna().any():
            self.df["BMI"] = self.df["BMI"].fillna(self.df["BMI"].median())

        return "Conditional and Numerical dependencies handled."

    def handle_ultrasound_features(self):
        all_features = self.grouped_features["Ultrasound"]
        primary_features = ["US_Number", "US_Performed", "Appendix_on_US"]
        secondary_features = list(set(all_features) - set(primary_features))

        def handle_primary_dependency(row, p_feature):
            us_num = row["US_Number"]
            us_perf = row["US_Performed"]
            appendix_on_us = row["Appendix_on_US"]

            us_num_missing = pd.isna(us_num) or us_num == ""
            appendix_missing = pd.isna(appendix_on_us) or appendix_on_us == ""
            us_perf_missing = pd.isna(us_perf) or us_perf == ""

            if p_feature == "US_Performed":
                if not us_num_missing:
                    return "yes"
                if us_num_missing and us_perf_missing:
                    return "no"
                return us_perf
            if feature == "US_Number":
                if us_perf == "no" and us_num_missing:
                    return "none"
                if us_perf == "yes" and us_num_missing:
                    return "missing"
                return us_num
            if p_feature == "Appendix_on_US":
                if appendix_missing and us_perf_missing:
                    return "missing"

                if appendix_missing and us_perf == "yes":
                    return "no"

            return row[feature]


        for feature in primary_features:
            self.df[feature] = self.df.apply(lambda row: handle_primary_dependency(row, feature), axis=1)

        for feature in secondary_features:
            self.df[feature] = self.df[feature].replace(to_replace=[np.nan, None, ""], value="not examined")

        return secondary_features




    def preparation(self, output_folder):
        self.drop_empty_rows()
        feature_groups = list(self.grouped_features)
        for feature_group in feature_groups:
            self.handle_features(feature_group)

        self.handle_dependent_features()
        self.handle_ultrasound_features()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.df.to_excel(os.path.join(output_folder, "prepared_data.xlsx"), index=False)


        return "Preparation done!"


