import os
import pandas as pd
import json


class HandleFeatures:
    def __init__(self, prev_stage, data_dir, excel_data, feature_info, derived_info, feature_groups):
        dataset = os.path.join(prev_stage, data_dir)
        self.df = pd.read_excel(os.path.join(dataset, excel_data))

        with open(os.path.join(dataset, feature_info), 'r') as f:
            self.feature_info = json.load(f)
        with open(os.path.join(dataset, derived_info), 'r') as f:
            self.derived_info = json.load(f)
        with open(os.path.join(dataset, feature_groups), 'r') as f:
            self.feature_groups = json.load(f)

        self.feature_value_map ={}

    def data_type_stabilization(self):
        missing_indicators = ['not examined', 'missing', 'nan', 'none', 'null', 'not observed']
        for feature, category in self.derived_info.items():
            info = self.feature_info.get(feature)
            if feature not in self.df.columns:
                continue

            if category in ["numeric", "numerically_dependent"]:
                self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')

            elif category == "bin_or_cat":
                if isinstance(info, dict) and "Binary" in info:
                    labels = info["Binary"]
                    if "yes" in labels:
                        mapping = {"yes": 1, "no": 0}
                    elif "male" in labels:
                        mapping = {"male": 1, "female": 0}
                    elif "appendicitis" in labels:
                        mapping = {"appendicitis": 1, "no appendicitis": 0}
                    elif "complicated" in labels:
                        mapping = {"complicated": 1, "uncomplicated": 0}
                    else:
                        mapping = {val: i for i, val in enumerate(labels)}

                    self.feature_value_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)

                if isinstance(info, dict) and "Categorical" in info:
                    labels = info["Categorical"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    self.feature_value_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)

            elif category == "ultrasound_feature":
                info = self.feature_info.get(feature)
                if info == "Continuous":
                    temp_series = self.df[feature].astype(str).str.lower().str.strip()
                    self.df[feature] = temp_series.replace(missing_indicators, -1)
                    self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')

                if isinstance(info, dict) and "Binary" in info:
                    labels = info["Binary"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    mapping.update({"not examined": -1})
                    self.feature_value_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping).fillna(-1)

                if isinstance(info, dict) and "Categorical" in info:
                    labels = info["Categorical"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    mapping.update({"not examined": -1})
                    self.feature_value_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping).fillna(-1)

                if info == "FreeText":
                    self.df[feature] = self.df[feature].replace('not examined', -1)
                    mapping = {"examined": "str", "not examined": -1}

                    self.feature_value_map.update({feature: mapping})



        print("Step 1 Success: Data types stabilized using derived_info categories.")
        return self.df

    def create_clinical_interactions(self):

        if 'WBC_Count' in self.df.columns and 'CRP' in self.df.columns:
            self.df['Inflammatory_Triage'] = self.df['WBC_Count'] * self.df['CRP']

        if 'Neutrophil_Percentage' in self.df.columns and 'WBC_Count' in self.df.columns:
            self.df['Left_Shift_Signal'] = self.df.apply(
                lambda row: row['Neutrophil_Percentage'] / row['WBC_Count']
                if row['WBC_Count'] > 0 else 0, axis=1
            )

        if 'Migratory_Pain' in self.df.columns and 'Lower_Right_Abd_Pain' in self.df.columns:
            self.df['Classic_Presentation_Flag'] = (
                    (self.df['Migratory_Pain'] == 1) &
                    (self.df['Lower_Right_Abd_Pain'] == 1)
            ).astype(int)

        print("Step 2 Success: Inflammatory Triage, Left Shift, and Classic Presentation Flag created.")
        return self.df

    def apply_medical_thresholds(self):
        if 'Appendix_Diameter' in self.df.columns:
            self.df['Pathological_Diameter'] = (
                (self.df['Appendix_Diameter'] >= 7.0)
            ).astype(int)

        if 'Body_Temperature' in self.df.columns:
            self.df['Fever_Flag'] = (
                (self.df['Body_Temperature'] >= 38.0)
            ).astype(int)

        if 'CRP' in self.df.columns:
            self.df['High_CRP_Flag'] = (
                (self.df['CRP'] >= 10.0)
            ).astype(int)

        print("Step 3 Success: Pathological Diameter, Fever, and High CRP flags created.")
        return self.df

    def extract_us_signals(self):
        self.df['Has_Images'] = self.df['US_Number'].apply(
            lambda x: 0 if str(x).lower() in ['missing', 'nan', 'none', ''] else 1
        )

        self.df['US_Sequence_Count'] = self.df['US_Number'].apply(
            lambda x: 0 if str(x).lower() in ['missing', 'nan', 'none', '']
            else len([i for i in str(x).split(',') if i.strip()])
        )

        secondary_features = [
            'Free_Fluids', 'Target_Sign', 'Appendicolith', 'Perfusion', 'Perforation',
            'Surrounding_Tissue_Reaction', 'Appendicular_Abscess', 'Pathological_Lymph_Nodes',
            'Bowel_Wall_Thickening', 'Ileus', 'Enteritis'
        ]

        self.df['Secondary_Findings_Score'] = self.df[
            [f for f in secondary_features if f in self.df.columns]
        ].apply(lambda row: (row >= 1).sum(), axis=1)

        print("Step 4 Success: Metadata and Signal Intensity features created.")
        return self.df

    def apply_categorical_encoding(self):
        all_categorical_features = [feature for feature, value in self.feature_info.items()
                                    if isinstance(value, dict) and "Categorical" in value]
        us_categorical_features = [feature for feature, value in self.derived_info.items()
                                   if value == "ultrasound_feature"]

        ohe_categorical_features = list(set(all_categorical_features) - set(us_categorical_features))


        for feature in ohe_categorical_features:
            if feature in self.df.columns:
                mapping = self.feature_value_map.get(feature, {})
                reverse_map = {v: k for k, v in mapping.items() if isinstance(v, (int, float))}
                self.df[feature] = self.df[feature].map(reverse_map).fillna(self.df[feature])

                self.df = pd.get_dummies(
                    self.df,
                    columns=[feature],
                    prefix=feature,
                    prefix_sep=':',
                    dtype=int
                )
        print("Step 5 Success: One-Hot Encoding applied to multiclass clinical features.")
        return self.df

    def remove_leakage_features(self):
        leakage_list = ["Length_of_Stay", "Management"]
        if "Management" not in leakage_list:
            leakage_list.append("Management")
        if "Length_of_Stay" not in leakage_list:
            leakage_list.append("Length_of_Stay")

        cols_to_drop = []
        for feat in leakage_list:
            if feat in self.df.columns:
                cols_to_drop.append(feat)
            cols_to_drop.extend([c for c in self.df.columns if c.startswith(f"{feat}:")])

        self.df.drop(columns=list(set(cols_to_drop)), errors='ignore', inplace=True)
        print(f"Step 6 Success: Leakage removed ({leakage_list}).")
        return self.df

    def align_feature_groups(self):
        updated_groups = {}

        current_columns = self.df.columns.tolist()

        for group_name, features in self.feature_groups.items():
            new_group_list = []
            for feat in features:
                if feat in current_columns:
                    new_group_list.append(feat)

                ohe_cols = [c for c in current_columns if c.startswith(f"{feat}:")]
                new_group_list.extend(ohe_cols)

            updated_groups[group_name] = list(set(new_group_list))

        engineered_list = [
            'Inflammatory_Triage', 'Left_Shift_Signal', 'Classic_Presentation_Flag',
            'Pathological_Diameter', 'Fever_Flag', 'High_CRP_Flag',
            'Has_Images', 'US_Sequence_Count', 'Secondary_Findings_Score'
        ]

        actual_engineered = [f for f in engineered_list if f in current_columns]

        if "engineered_features" not in updated_groups:
            updated_groups["engineered_features"] = actual_engineered
        else:
            updated_groups["engineered_features"].extend(actual_engineered)
            updated_groups["engineered_features"] = list(set(updated_groups["engineered_features"]))

        self.feature_groups = updated_groups
        print("Step 7 Success: Feature groups aligned for Stage 6 Importance analysis.")
        return self.feature_groups

    def save_data(self, output_folder):
        self.data_type_stabilization()
        self.create_clinical_interactions()
        self.apply_medical_thresholds()
        self.extract_us_signals()
        self.apply_categorical_encoding()
        self.remove_leakage_features()
        self.align_feature_groups()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        self.df.to_excel(os.path.join(output_folder, "engineered_data.xlsx"), index=False)

        encoding_map_output_path = os.path.join(output_folder, "encoding_map.json")
        feature_groups_path = os.path.join(output_folder, "engineered_feature_groups.json")

        with open(encoding_map_output_path, "w") as f:
            json.dump(self.feature_value_map, f, indent=4)

        with open(feature_groups_path, "w") as f:
            json.dump(self.feature_groups, f, indent=4)


        return 0
