# sample_data.py inside the package
import duckdb
from importlib import resources
import tempfile
import shutil
import atexit
from pathlib import Path

class SampleData:
    def __init__(self):
        # Access the data files through the package resources API
        self.data_files = resources.files('traffic_anomaly').joinpath('data')
        self._temp_dir = None
        self._vehicle_counts = None
        self._travel_times = None
        self._changepoints_input = None
        self._connectivity = None
        
        # Register cleanup to run at exit
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Clean up the temporary directory on exit."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                # Best effort cleanup, suppress errors during exit
                pass

    def _get_temp_dir(self):
        """Lazy creation of temp directory."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix='traffic_anomaly_')
        return Path(self._temp_dir)

    def _load_dataset(self, filename):
        """Helper to safely load a dataset from resources into a temp file."""
        temp_dir = self._get_temp_dir()
        target_path = temp_dir / filename
        
        # Only copy if it doesn't already exist (though in a new temp dir it won't)
        if not target_path.exists():
            with resources.as_file(self.data_files.joinpath(filename)) as source_path:
                shutil.copy2(source_path, target_path)
                
        # Load with DuckDB from the writable temp path
        return duckdb.sql(f"select * from '{target_path}'").df()

    @property
    def vehicle_counts(self):
        if self._vehicle_counts is None:
            self._vehicle_counts = self._load_dataset('sample_counts.parquet')
        return self._vehicle_counts

    @property
    def travel_times(self):
        if self._travel_times is None:
            self._travel_times = self._load_dataset('sample_travel_times.parquet')
        return self._travel_times

    @property
    def changepoints_input(self):
        if self._changepoints_input is None:
            self._changepoints_input = self._load_dataset('sample_changepoint_input.parquet')
        return self._changepoints_input

    @property
    def connectivity(self):
        if self._connectivity is None:
            self._connectivity = self._load_dataset('sample_connectivity.parquet')
        return self._connectivity

# Create an instance of the class
sample_data = SampleData()
