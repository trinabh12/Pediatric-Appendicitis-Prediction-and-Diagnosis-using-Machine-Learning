import os
import json
import re
import pandas as pd
import shutil


class ValidationAndProfiling:

    def __init__(self, prev_stage, dataset_dir, file_name, data_info, data_summary, data_grouped, img_data):
        self.dataset = os.path.join(prev_stage, dataset_dir)
        self.file_path = os.path.join(self.dataset, file_name)
        self.raw_data = pd.read_excel(self.file_path)
        self.img_data_path = os.path.join(prev_stage, img_data)

        info_path = os.path.join(self.dataset, data_info)
        summary_path = os.path.join(self.dataset, data_summary)
        grouped_path = os.path.join(self.dataset, data_grouped)
        self.feature_info = json.load(open(info_path, "r"))
        self.feature_summary = json.load(open(summary_path, "r"))
        self.feature_grouped = json.load(open(grouped_path, "r"))

        self.output_xlsx = "validated_data.xlsx"
        self.output_missing = "missing_data.json"
        self.output_validation = "validated_feature_info.json"
        self.output_grouped = "validated_feature_grouped.json"

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

        info_stats = get_aggregated_counts(self.feature_info)
        summary_stats = get_aggregated_counts(self.feature_summary)


        if info_stats == summary_stats:
            self.validated_info = self.feature_info
            return True
        else:
            return "Feature Info and Feature Summary don't match"

    def bridge_map(self):
        def get_type(val):
            if isinstance(val, dict):
                return list(val.keys())[0]
            return str(val)

        def normalize(text):
            text = text.lower()
            text = text.replace('ae', 'e').replace('tnderness', 'tenderness')
            text = text.replace('gynaecological', 'gynecological')
            text = text.replace('erythrocytes', 'rbc').replace('white blood cells', 'wbc')
            text = text.replace('migration of pain', 'migratory pain')
            return re.sub(r'[^a-z0-9]', '', text)

        aliases = {
            "Visibility of appendix": "Appendix_on_US",
            "Anorexia": "Loss_of_Appetite",
            "Cough tenderness": "Coughing_Pain",
            "Tenderness in right lower quadrant (RLQ)": "Lower_Right_Abd_Pain",
            "Ultrasound images": "US_Number",
            "Performed ultrasound (US)": "US_Performed",
            "Free intraperitoneal fluid": "Free_Fluids",
            "Appendix layer structure": "Appendix_Wall_Layers",
            "Pediatric appendicitis score (PAS), pts": "Paedriatic_Appendicitis_Score",
            "Length of stay, days": "Length_of_Stay",
            "Thrombocyte count, /nl": "Thrombocyte_Count",
            "Appendicolith": "Appendicolith",
            "Perforation": "Perforation",
            "Appendicular abscess": "Appendicular_Abscess",
            "Presumptive diagnosis": "Diagnosis_Presumptive",
            "Location of abscess": "Abscess_Location",
            "Location of pathological lymph nodes": "Lymph_Nodes_Location",
            "Thickening of the bowel wall": "Bowel_Wall_Thickening",
            "Neutrophils, %": "Neutrophil_Percentage",
            "Migration of pain": "Migratory_Pain"
        }

        bridge_map = {}

        for l_name, s_name in aliases.items():
            if l_name in self.feature_summary and s_name in self.feature_info:
                bridge_map[l_name] = s_name

        for long_name, summary_val in self.feature_summary.items():
            if long_name in bridge_map:
                continue

            s_type = get_type(summary_val)
            norm_long = normalize(long_name)

            for short_name, info_val in self.feature_info.items():
                if short_name in bridge_map.values():
                    continue

                i_type = get_type(info_val)

                if s_type != i_type:
                    if not (s_type in ["FreeText", "Discrete", "Image BMP"] or i_type in ["FreeText", "Discrete",
                                                                                          "Image BMP"]):
                        continue

                norm_short = normalize(short_name)

                if norm_short in norm_long or norm_long in norm_short:
                    bridge_map[long_name] = short_name
                    break

                brackets = re.findall(r'\((.*?)\)', long_name)
                if brackets and any(normalize(b) in norm_short for b in brackets):
                    bridge_map[long_name] = short_name
                    break

        return bridge_map

    def update_grouped_features(self):
        mapping = self.bridge_map()
        updated_grouped = {}

        for category, feature_list in self.feature_grouped.items():
            translated_list = [mapping.get(summary_feature) for summary_feature in feature_list]
            updated_grouped[category] = translated_list

        return updated_grouped

    def validate_image_data(self):
        """
        Analyzes image names to derive the format and check for anomalies.
        Returns a summary of the image naming structure.
        """
        if not os.path.exists(self.img_data_path):
            return {"error": "Image path not found"}

        img_files = [f for f in os.listdir(self.img_data_path)
                     if os.path.isfile(os.path.join(self.img_data_path, f))]

        # Regex: starts with digits, a dot, then digits (e.g., 101.1)
        standard_pattern = re.compile(r"^\d+\.\d+")

        valid_files = []
        anomalies = []
        formats = {}

        for f in img_files:
            f = str(f)
            if standard_pattern.match(f):
                valid_files.append(f)
                # Derive format by counting segments split by dots
                # e.g., '9.1 Appendix.bmp' -> 3 segments
                num_segments = len(f.split('.'))
                fmt_tag = f"{num_segments}-segment-dot-notation"
                formats[fmt_tag] = formats.get(fmt_tag, 0) + 1
            else:
                anomalies.append(f)

        validation_summary = {
            "status": "Success" if not anomalies else "Warnings Found",
            "total_images": len(img_files),
            "naming_format": "SubjectID.ViewID [Description].ext",
            "format_distribution": formats,
            "anomaly_count": len(anomalies),
            "anomaly_list": anomalies
        }

        return validation_summary

    def save_tab_report(self, output_folder: str):
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

        grouped_path = os.path.join(output_folder, self.output_grouped)
        with open(grouped_path, 'w') as f:
            json.dump(self.update_grouped_features(), f, indent=4)

        return 0

    def save_img_report(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

        img_files = [f for f in os.listdir(self.img_data_path)
                     if self.img_data_path]

        print(f"Copying {len(img_files)} images to validation folder...")

        for filename in img_files:
            src = os.path.join(self.img_data_path, str(filename))
            dst = os.path.join(output_folder, str(filename))
            shutil.copy2(src, dst)  # copy2 preserves metadata

        # 3. Save the JSON metadata report
        report_data = {
            "validation_stage": "Stage 2",
            "source_path": self.img_data_path,
            "destination_path": output_folder,
            "total_files_processed": len(img_files),
            "files": sorted(img_files)
        }

        report_path = os.path.join(output_folder, "image_validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        print(f"Image validation report and physical copy completed at: {output_folder}")
        return 0
