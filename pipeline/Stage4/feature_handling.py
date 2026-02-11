import os
import pandas as pd
import numpy as np
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

        self.encoding_map ={}

    def data_type_stabilization(self):
        missing_indicators = ['not examined', 'missing', 'nan', 'none', 'null']
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

                    self.encoding_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)

                if isinstance(info, dict) and "Categorical" in info:
                    labels = info["Categorical"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    self.encoding_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)

            elif category == "ultrasound_feature":
                info = self.feature_info.get(feature)
                if info == "Continuous":
                    if isinstance(self.df[feature], str) and self.feature_info[feature] in missing_indicators:
                        continue

                if isinstance(info, dict) and "Binary" in info:
                    labels = info["Binary"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    self.encoding_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)

                if isinstance(info, dict) and "Categorical" in info:
                    labels = info["Categorical"]
                    mapping = {val: i for i, val in enumerate(labels)}
                    self.encoding_map.update({feature: mapping})
                    self.df[feature] = self.df[feature].astype(str).str.lower().str.strip().map(mapping)



        print("Success: Data types stabilized using derived_info categories.")
        return self.df

    def save_data(self, output_folder):
        self.data_type_stabilization()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        self.df.to_excel(os.path.join(output_folder, "engineered_data.xlsx"), index=False)

        encoding_map_output_path = os.path.join(output_folder, "encoding_map.json")
        print(len(list(self.encoding_map)))

        with open(encoding_map_output_path, "w") as f:
            json.dump(self.encoding_map, f, indent=4)
