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

    def find_conditional_dependencies(self, missing_threshold=0.8, diff_threshold=0.5):
        df = self.df
        potential_gatekeepers =[k for k, v in self.feature_info.items() if isinstance(v, dict) and "Binary" in v]

        dependency_map = {}

        for gate in potential_gatekeepers:
            if gate not in df.columns:
                continue

            for state in df[gate].dropna().unique():
                neg_mask = (df[gate] == state)
                pos_mask = (df[gate] != state)

                if np.sum(neg_mask) < 5 or np.sum(pos_mask) < 5:
                    continue

                dependents = []
                for column in df.columns:
                    if column == gate or column in potential_gatekeepers:
                        continue

                    m_neg = df.loc[neg_mask, column].isna().mean()
                    m_pos = df.loc[pos_mask, column].isna().mean()

                    if m_neg > missing_threshold and (m_neg - m_pos) > diff_threshold:
                        dependents.append(column)

                    if dependents:
                        key = f"{gate} (if '{state}')"
                        dependency_map[key] = dependents

        return dependency_map
