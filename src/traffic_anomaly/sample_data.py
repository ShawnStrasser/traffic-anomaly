# sample_data.py inside the package
import duckdb
from importlib import resources

class SampleData:
    def __init__(self):
        # Access the data files through the package resources API
        data_path = resources.files('traffic_anomaly').joinpath('data')

        self.vehicle_counts = duckdb.sql(f"select * from '{data_path.joinpath('sample_counts.parquet')}'").df()
        self.travel_times = duckdb.sql(f"select * from '{data_path.joinpath('sample_travel_times.parquet')}'").df()
        self.changepoints_input = duckdb.sql(f"select * from '{data_path.joinpath('sample_changepoint_input.parquet')}'").df()
        self.connectivity = duckdb.sql(f"select * from '{data_path.joinpath('sample_connectivity.parquet')}'").df()

# Create an instance of the class
sample_data = SampleData()