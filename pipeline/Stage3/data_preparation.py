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

        self.total_rows = len(self.df)
        self.branches = {
            "trunk": [],
            "conditional": [],
            "sparse": []
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


    def find_all_dependencies(self, missing_threshold=0.8, diff_threshold=0.4):
        df = self.df
        dependency_map = {
            "conditional": {},
            "calculated": {}
        }

        potential_gates = [k for k, v in self.feature_info.items() if isinstance(v, dict) and "Binary" in v]

        for gate in potential_gates:
            if gate not in df.columns:
                continue
            for state in df[gate].dropna().unique():
                neg_mask = (df[gate] == state)
                pos_mask = (df[gate] != state)
                if np.sum(neg_mask) < 5 or np.sum(pos_mask) < 5:
                    continue

                dependents = []
                for col in df.columns:
                    if col == gate or col in potential_gates:
                        continue
                    m_neg = df.loc[neg_mask, col].isna().mean()
                    m_pos = df.loc[pos_mask, col].isna().mean()

                    if m_neg > missing_threshold and (m_neg - m_pos) > diff_threshold:
                        dependents.append(col)

                if dependents:
                    dependency_map["conditional"][f"{gate}=={state}"] = dependents

        numeric_cols = [k for k, v in self.feature_info.items() if v in ["Continuous", "Discrete"]]

        for parent in numeric_cols:
            if parent not in df.columns:
                continue
            parent_missing_mask = df[parent].isna()
            if parent_missing_mask.sum() == 0:
                continue

            for child in numeric_cols:
                if parent == child:
                    continue

                child_missing_when_parent_missing = df.loc[parent_missing_mask, child].isna().all()

                if child_missing_when_parent_missing:
                    if parent not in dependency_map["calculated"]:
                        dependency_map["calculated"][parent] = []
                    dependency_map["calculated"][parent].append(child)

        return dependency_map
