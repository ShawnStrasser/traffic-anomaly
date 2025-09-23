# sample_data.py inside the package
import duckdb
from importlib import resources

class SampleData:
    def __init__(self):
        # Access the data files through the package resources API
        data_files = resources.files('traffic_anomaly').joinpath('data')

        # Use the 'as_file' context manager to extract each file
        # and get a real filesystem path that DuckDB can read.
        with resources.as_file(data_files.joinpath('sample_counts.parquet')) as p:
            self.vehicle_counts = duckdb.sql(f"select * from '{p}'").df()

        with resources.as_file(data_files.joinpath('sample_travel_times.parquet')) as p:
            self.travel_times = duckdb.sql(f"select * from '{p}'").df()

        with resources.as_file(data_files.joinpath('sample_changepoint_input.parquet')) as p:
            self.changepoints_input = duckdb.sql(f"select * from '{p}'").df()

        with resources.as_file(data_files.joinpath('sample_connectivity.parquet')) as p:
            self.connectivity = duckdb.sql(f"select * from '{p}'").df()
# Create an instance of the class
sample_data = SampleData()