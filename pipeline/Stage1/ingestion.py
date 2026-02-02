import os
import pandas as pd
import json
import ast


class IngestionStage:

    def __init__(self, root_dir, dataset_dir, file_name):
        self.dataset_dir = os.path.join(root_dir, dataset_dir)
        self.file = file_name
        self.output_data = "raw_data.xlsx"
        self.output_summary = "feature_summary.json"
        self.output_info = "feature_info.json"
        self.file_path = os.path.join(self.dataset_dir, self.file)

        self.df_cases = pd.read_excel(self.file_path, sheet_name="All cases")
        self.df_summary = pd.read_excel(self.file_path, sheet_name="Data Summary")

    def feature_summary(self):
        summary = {}

        for _, row in self.df_summary.iterrows():
            feature_name = str(row.iloc[0]).strip()  # 3rd column has variable names corresponding to the dataset
            raw_info = str(row.iloc[-1]).lower()  # last column has the value types

            if "binary" in raw_info:
                feature_type = {"Binary": raw_info.replace("binary:", "").split("/")}

            elif "categorical" in raw_info:
                feature_type = {"Categorical": raw_info.replace("categorical:", "")[3:].replace("\n·", "|").split("|")}
                # sliced everything before the 4th character to remove unnecessary symbols
            elif "continuous" in raw_info:
                feature_type = "Continuous"
            elif "discrete" in raw_info:
                feature_type = "Discrete"
            elif "free" in raw_info:
                feature_type = "FreeText"
            elif "images" in raw_info:
                feature_type = "Image BMP"
            else:
                feature_type = "Unknown"

            summary[feature_name] = feature_type

        return summary

    def feature_info(self):
        def infer_semantic_type(series, series_name):
            values = (series.dropna().astype(str).str.strip().str.lower().unique())
            unique_count = len(values)
            if unique_count == 0:
                return "empty", []
            elif unique_count == 1:
                return "constant"
            elif unique_count == 2:
                return "Binary", values.tolist()
            elif 2 < unique_count < 7:
                return "Categorical", values.tolist()

            img_txt = "us_number"
            if img_txt in series_name:
                return "Image BMP"

            else:
                try:
                    numeric = series.dropna().astype(float).tolist()
                    numeric_quotient = [x // 1 for x in numeric]

                    if sum(numeric_quotient) == sum(numeric):
                        return "Discrete"
                    else:
                        return "Continuous"

                except:
                    return "FreeText"

        def str_tuple_to_dict(string_tuple):
            data_str = string_tuple
            data_tuple = ast.literal_eval(data_str)

            # Convert tuple to a dictionary
            # data_tuple[0] is the key, data_tuple[1] is the list
            result = {data_tuple[0]: data_tuple[1]}

            return result
            # Output: {'Binary': ['female', 'male']}

        info = {}

        for feature in self.df_cases.columns:
            rows = self.df_cases[feature]
            info[feature.strip()] = str(infer_semantic_type(rows, feature.lower()))

            if "Binary" in info[feature.strip()]:
                info[feature.strip()] = str_tuple_to_dict(info[feature.strip()])
            elif "Categorical" in info[feature.strip()]:
                info[feature.strip()] = str_tuple_to_dict(info[feature.strip()])


        return info

    def extract_xlsx(self, output_folder: str):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found:{self.file_path}")
        else:
            os.makedirs(output_folder, exist_ok=True)

            cases_path = os.path.join(output_folder, self.output_data)
            self.df_cases.to_excel(excel_writer=cases_path, index=False)

            summary_dict = self.feature_summary()
            summary_path = os.path.join(output_folder, self.output_summary)
            with open(summary_path, "w") as f:
                json.dump(summary_dict, f, indent=4)

            info_dict = self.feature_info()
            info_path = os.path.join(output_folder, self.output_info)
            with open(info_path, "w") as f:
                json.dump(info_dict, f, indent=4)

        return 0
