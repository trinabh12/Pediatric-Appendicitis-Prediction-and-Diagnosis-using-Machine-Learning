import os
import json
import pandas as pd


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

        self.logic_dependencies = {
            # Clinical
            "Migratory_Pain": "Lower_Right_Abd_Pain",
            "Contralateral_Rebound_Tenderness": "Lower_Right_Abd_Pain",
            "Ipsilateral_Rebound_Tenderness": "Lower_Right_Abd_Pain",
            "Coughing_Pain": "Peritonitis",
            # Ultrasound
            "Appendix_Diameter": "Appendix_on_US",
            "Appendix_Wall_Layers": "Appendix_on_US",
            "Target_Sign": "Appendix_on_US",
            "Perfusion": "Appendix_on_US",
            "Appendicolith": "Appendix_on_US",
            "Perforation": "Appendix_on_US",
            "Abscess_Location": "Appendicular_Abscess",
            "Lymph_Nodes_Location": "Pathological_Lymph_Nodes"
        }

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
        conditional_dependencies = ["Migratory_Pain", "Contralateral_Rebound_Tenderness",
                                    "Ipsilateral_Rebound_Tenderness", "Coughing_Pain", "Appendix_Diameter",
                                    "Appendix_Wall_Layers", "Target_Sign", "Perfusion", "Appendicolith", "Perforation",
                                    "Abscess_Location", "Lymph_Nodes_Location"]
        if feature_name in numeric_dependencies:
            return "numerically_dependent"
        if feature_name in conditional_dependencies:
            return "conditionally_dependent"
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

        if 50 >= missing_pct > 5:
            self.df[feature_name] = self.df[feature_name].fillna(0)
            return f"{feature_name}: Filled null values with 0."

        if missing_pct <= 5:
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

    def encode_binary_categorical(self, feature):
        info = self.feature_info.get(feature)
        options = info.get("Binary") or info.get("Categorical")

        if "yes" in options and "no" in options:
            mapping = {"no": 0, "yes": 1}
        else:
            mapping = {val: i for i, val in enumerate(options)}

        mapping_report = {feature: mapping}

        self.df[feature] = self.df[feature].map(mapping)
        return mapping_report


    def freetext_and_imginfo_filling(self, feature_name):
        if self.feature_value_type(feature_name) != "freetext":
            return f"{feature_name} is not a freetext feature."
        if self.feature_value_type(feature_name) != "BMP":
            return f"{feature_name} is not a US image feature."

        fill_value = "none"
        self.df[feature_name] = self.df[feature_name].fillna(fill_value)
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

        for child, parent in self.logic_dependencies.items():
            if child in self.df.columns and parent in self.df.columns:
                is_no = self.df[parent].astype(str).str.lower().str.strip() == 'no'
                mask = self.df[child].isna() & is_no
                self.df.loc[mask, child] = 0

                remaining_mask = self.df[child].isna()
                if remaining_mask.any():
                    v_type = self.feature_value_type(child)
                    if v_type == "numeric":
                        self.numeric_filling(child)
                    else:
                        self.binary_and_categorical_filling(child)

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

    def preparation(self, output_folder):
        self.drop_empty_rows()
        feature_groups = list(self.grouped_features)
        for feature_group in feature_groups:
            self.handle_features(feature_group)

        self.handle_dependent_features()

        self.df = self.df.fillna("None")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.df.to_excel(os.path.join(output_folder, "prepared_data.xlsx"), index=False)

        return "Preparation done!"


