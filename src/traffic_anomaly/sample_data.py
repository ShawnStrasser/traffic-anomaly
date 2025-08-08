# sample_data.py inside the package
import pandas as pd
import os

# Assuming this file is in the same directory as the `data` directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')

class SampleData:
    def __init__(self):
        self.vehicle_counts = pd.read_parquet(os.path.join(data_dir, 'sample_counts.parquet'))
        self.travel_times = pd.read_parquet(os.path.join(data_dir, 'sample_travel_times.parquet'))
        self.changepoints_input = pd.read_parquet(os.path.join(data_dir, 'sample_changepoint_input.parquet'))

# Create an instance of the class
sample_data = SampleData()