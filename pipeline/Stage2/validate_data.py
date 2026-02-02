import os
import json
import pandas as pd


class ValidationAndProfiling:

    def __init__(self, prev_stage, dataset_dir, file_name, data_info, data_summary):
        self.dataset = os.path.join(prev_stage, dataset_dir)
        self.file_path = os.path.join(self.dataset, file_name)
        self.raw_data = pd.read_excel(self.file_path)
        self.info_path = os.path.join(self.dataset, data_info)
        self.summary_path = os.path.join(self.dataset, data_summary)

        self.output_xlsx = "validated_data.xlsx"
        self.output_missing = "missing_data.json"
        self.output_validation = "validated_feature_info.json"

        self.validated_info = {}



        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.df = pd.read_excel(self.file_path)

    def missing_summary(self) -> dict:
        missing = {}
        for column in self.df.columns:
            missing[column] = int(self.df[column].isna().sum())
        return missing

    def validation(self):
        feature_info = json.load(open(self.info_path, "r"))
        feature_summary = json.load(open(self.summary_path, "r"))


        def get_aggregated_counts(data):
            counts = {"binary_categorical": 0,
                      "discrete_continuous": 0,
                      "image": 0,
                      "freetext": 0}

            for value in data.values():

                if isinstance(value, dict):
                    if "Binary" in value or "Categorical" in value:
                        counts["binary_categorical"] +=1

                elif isinstance(value, str):
                    if value in ["Discrete", "Continuous"]:
                        counts["discrete_continuous"] +=1

                    elif value == "Image BMP":
                        counts["image"] +=1

                    elif value == "FreeText":
                        counts["freetext"] +=1
            return counts

        info_stats = get_aggregated_counts(feature_info)
        summary_stats = get_aggregated_counts(feature_summary)


        if info_stats == summary_stats:
            self.validated_info = feature_info
            return True
        else:
            return "Feature Info and Feature Summary don't match"


    def save_report(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        missing_dict = self.missing_summary()

        missing_output_path = os.path.join(output_folder, self.output_missing)
        validation_output_path = os.path.join(output_folder, self.output_validation)

        with open(missing_output_path, "w") as f:
            json.dump(missing_dict, f, indent=4)

        with open(validation_output_path, "w") as f:
            json.dump(self.validated_info, f, indent=4)

        cases_path = os.path.join(output_folder, self.output_xlsx)
        self.raw_data.to_excel(excel_writer=cases_path, index=False)

        return 0


