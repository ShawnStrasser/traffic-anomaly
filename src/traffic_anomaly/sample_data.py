# sample_data.py inside the package
import pandas as pd
import os

# Assuming this file is in the same directory as the `data` directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')

class SampleData:
    def __init__(self):
        self.data = pd.read_parquet(os.path.join(data_dir, 'sample_data.parquet'))
        self.groups = pd.read_parquet(os.path.join(data_dir, 'sample_data_groups.parquet'))

# Create an instance of the class
sample_data = SampleData()