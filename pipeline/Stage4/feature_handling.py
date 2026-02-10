import os
import pandas as pd
import numpy as np


class HandleFeatures:
    def __init__(self, datadir, excel_data):
        self.df = pd.read_excel(os.path.join(datadir, excel_data))


