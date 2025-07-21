"""
Unit tests for traffic_anomaly package.

This test suite compares calculated results with precalculated reference data
and verifies version consistency between __init__.py and pyproject.toml.
"""

import unittest
import pandas as pd
import os
import sys
import toml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import traffic_anomaly
    from traffic_anomaly import sample_data
    PACKAGE_AVAILABLE = True
except ImportError as e:
    PACKAGE_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestVersionConsistency(unittest.TestCase):
    """Test that version numbers match between __init__.py and pyproject.toml"""
    
    def test_version_consistency(self):
        """Verify that __init__.py version matches pyproject.toml version"""
        # Read version from pyproject.toml
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
        
        pyproject_version = pyproject_data['project']['version']
        
        # Read version from __init__.py
        if PACKAGE_AVAILABLE:
            init_version = traffic_anomaly.__version__
        else:
            # Fallback: read directly from file if import fails
            init_path = project_root / "src" / "traffic_anomaly" / "__init__.py"
            with open(init_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('__version__'):
                        init_version = line.split('=')[1].strip().strip('\'"')
                        break
        
        self.assertEqual(init_version, pyproject_version, 
                        f"Version mismatch: __init__.py has {init_version}, "
                        f"pyproject.toml has {pyproject_version}")


@unittest.skipIf(not PACKAGE_AVAILABLE, f"Package not available: {IMPORT_ERROR if not PACKAGE_AVAILABLE else ''}")
class TestDecomposition(unittest.TestCase):
    """Test median decomposition functionality"""
    
    def setUp(self):
        """Set up test data and reference paths"""
        self.test_data_dir = Path(__file__).parent / "precalculated"
        self.travel_times = sample_data.travel_times
        self.vehicle_counts = sample_data.vehicle_counts
    
    def test_median_decompose_travel_times(self):
        """Test median decomposition with travel times data matches precalculated results"""
        # Perform decomposition
        decomp = traffic_anomaly.median_decompose(
            data=self.travel_times,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id', 'group'],
            freq_minutes=60,
            rolling_window_days=7,
            drop_days=7,
            min_rolling_window_samples=56,
            min_time_of_day_samples=7,
            drop_extras=False,
            to_sql=False
        )
        
        # Load expected results
        expected_path = self.test_data_dir / "test_decomp.parquet"
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self.assertEqual(len(decomp), len(expected), 
                        "Decomposition result has different number of rows than expected")
        
        # Check that all expected columns are present
        expected_columns = set(expected.columns)
        actual_columns = set(decomp.columns)
        self.assertEqual(expected_columns, actual_columns,
                        f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
        
        # Sort both dataframes for comparison
        sort_columns = ['timestamp', 'id'] if 'id' in decomp.columns else ['timestamp']
        decomp_sorted = decomp.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare numerical columns with tolerance
        numerical_columns = decomp.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            if col in expected.columns:
                pd.testing.assert_series_equal(
                    decomp_sorted[col], expected_sorted[col],
                    check_names=True, check_dtype=False,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Mismatch in column {col}"
                )
    
    def test_median_decompose_vehicle_counts(self):
        """Test median decomposition with vehicle counts data matches precalculated results"""
        # Perform decomposition
        decomp2 = traffic_anomaly.median_decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        
        # Load expected results
        expected_path = self.test_data_dir / "test_decomp2.parquet"
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self.assertEqual(len(decomp2), len(expected), 
                        "Decomposition result has different number of rows than expected")
        
        # Check that all expected columns are present
        expected_columns = set(expected.columns)
        actual_columns = set(decomp2.columns)
        self.assertEqual(expected_columns, actual_columns,
                        f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
        
        # Sort both dataframes for comparison
        sort_columns = ['timestamp', 'intersection', 'detector']
        decomp2_sorted = decomp2.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare numerical columns with tolerance
        numerical_columns = decomp2.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            if col in expected.columns:
                pd.testing.assert_series_equal(
                    decomp2_sorted[col], expected_sorted[col],
                    check_names=True, check_dtype=False,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Mismatch in column {col}"
                )


@unittest.skipIf(not PACKAGE_AVAILABLE, f"Package not available: {IMPORT_ERROR if not PACKAGE_AVAILABLE else ''}")
class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality"""
    
    def setUp(self):
        """Set up test data and reference paths"""
        self.test_data_dir = Path(__file__).parent / "precalculated"
        self.travel_times = sample_data.travel_times
        self.vehicle_counts = sample_data.vehicle_counts
        
        # Pre-compute decompositions for anomaly detection tests
        self.decomp = traffic_anomaly.median_decompose(
            data=self.travel_times,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id', 'group'],
            freq_minutes=60,
            rolling_window_days=7,
            drop_days=7,
            min_rolling_window_samples=56,
            min_time_of_day_samples=7,
            drop_extras=False,
            to_sql=False
        )
        
        self.decomp2 = traffic_anomaly.median_decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
    
    def test_find_anomaly_basic(self):
        """Test basic anomaly detection matches precalculated results"""
        # Perform anomaly detection
        anomaly = traffic_anomaly.find_anomaly(
            decomposed_data=self.decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5
        )
        
        # Load expected results
        expected_path = self.test_data_dir / "test_anomaly.parquet"
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self.assertEqual(len(anomaly), len(expected), 
                        "Anomaly result has different number of rows than expected")
        
        # Check that all expected columns are present
        expected_columns = set(expected.columns)
        actual_columns = set(anomaly.columns)
        self.assertEqual(expected_columns, actual_columns,
                        f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
        
        # Sort both dataframes for comparison
        sort_columns = ['timestamp', 'id']
        anomaly_sorted = anomaly.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare numerical columns with tolerance
        numerical_columns = anomaly.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            if col in expected.columns:
                pd.testing.assert_series_equal(
                    anomaly_sorted[col], expected_sorted[col],
                    check_names=True, check_dtype=False,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Mismatch in column {col}"
                )
    
    def test_find_anomaly_with_mad(self):
        """Test anomaly detection with MAD option matches precalculated results"""
        # Perform anomaly detection with MAD
        anomaly2 = traffic_anomaly.find_anomaly(
            decomposed_data=self.decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5,
            group_grouping_columns=['group'],
            MAD=True
        )
        
        # Load expected results
        expected_path = self.test_data_dir / "test_anomaly2.parquet"
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self.assertEqual(len(anomaly2), len(expected), 
                        "Anomaly result has different number of rows than expected")
        
        # Check that all expected columns are present
        expected_columns = set(expected.columns)
        actual_columns = set(anomaly2.columns)
        self.assertEqual(expected_columns, actual_columns,
                        f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
        
        # Sort both dataframes for comparison
        sort_columns = ['timestamp', 'id']
        anomaly2_sorted = anomaly2.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare numerical columns with tolerance
        numerical_columns = anomaly2.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            if col in expected.columns:
                pd.testing.assert_series_equal(
                    anomaly2_sorted[col], expected_sorted[col],
                    check_names=True, check_dtype=False,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Mismatch in column {col}"
                )
    
    def test_find_anomaly_with_geh(self):
        """Test anomaly detection with GEH statistic matches precalculated results"""
        # Perform anomaly detection with GEH
        anomaly3 = traffic_anomaly.find_anomaly(
            decomposed_data=self.decomp2,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
            return_sql=False
        )
        
        # Load expected results
        expected_path = self.test_data_dir / "test_anomaly3.parquet"
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self.assertEqual(len(anomaly3), len(expected), 
                        "Anomaly result has different number of rows than expected")
        
        # Check that all expected columns are present
        expected_columns = set(expected.columns)
        actual_columns = set(anomaly3.columns)
        self.assertEqual(expected_columns, actual_columns,
                        f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
        
        # Sort both dataframes for comparison
        sort_columns = ['timestamp', 'intersection', 'detector']
        anomaly3_sorted = anomaly3.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare numerical columns with tolerance
        numerical_columns = anomaly3.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            if col in expected.columns:
                pd.testing.assert_series_equal(
                    anomaly3_sorted[col], expected_sorted[col],
                    check_names=True, check_dtype=False,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Mismatch in column {col}"
                )


class TestPrecalculatedDataExists(unittest.TestCase):
    """Test that all expected precalculated data files exist"""
    
    def setUp(self):
        """Set up paths to test data"""
        self.test_data_dir = Path(__file__).parent / "precalculated"
        self.expected_files = [
            "test_decomp.parquet",
            "test_anomaly.parquet", 
            "test_anomaly2.parquet",
            "test_decomp2.parquet",
            "test_anomaly3.parquet"
        ]
    
    def test_precalculated_files_exist(self):
        """Verify all expected precalculated files exist"""
        for filename in self.expected_files:
            file_path = self.test_data_dir / filename
            self.assertTrue(file_path.exists(), 
                          f"Expected precalculated file {filename} not found at {file_path}")
    
    def test_precalculated_files_readable(self):
        """Verify all precalculated files can be read as parquet"""
        for filename in self.expected_files:
            file_path = self.test_data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    self.assertGreater(len(df), 0, f"File {filename} is empty")
                except Exception as e:
                    self.fail(f"Could not read {filename}: {e}")


if __name__ == '__main__':
    # Print information about package availability
    if not PACKAGE_AVAILABLE:
        print(f"Warning: traffic_anomaly package not available: {IMPORT_ERROR}")
        print("Only version consistency and file existence tests will run.")
    
    unittest.main(verbosity=2)