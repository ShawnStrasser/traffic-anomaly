# sample_data.py inside the package
import pandas as pd
from importlib import resources

class SampleData:
    def __init__(self):
        # Access the data files through the package resources API
        data_path = resources.files('traffic_anomaly').joinpath('data')
        
        self.vehicle_counts = pd.read_parquet(data_path.joinpath('sample_counts.parquet'))
        self.travel_times = pd.read_parquet(data_path.joinpath('sample_travel_times.parquet'))
        self.changepoints_input = pd.read_parquet(data_path.joinpath('sample_changepoint_input.parquet'))
        self.connectivity = pd.read_parquet(data_path.joinpath('sample_connectivity.parquet'))

# Create an instance of the class
sample_data = SampleData()