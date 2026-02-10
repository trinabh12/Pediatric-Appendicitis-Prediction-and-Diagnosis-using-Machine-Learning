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
        with open(os.path.join(dataset, feature_groups), 'r') as f:
            self.feature_groups = json.load(f)



