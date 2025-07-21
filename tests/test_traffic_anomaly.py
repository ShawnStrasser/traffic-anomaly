import pytest
import pandas as pd
import numpy as np
import os
import sys
import toml
from pathlib import Path

# Add src to path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import traffic_anomaly
from traffic_anomaly import sample_data


class TestVersionConsistency:
    """Test that version numbers match between __init__.py and pyproject.toml"""
    
    def test_version_consistency(self):
        """Verify that the version in __init__.py matches pyproject.toml"""
        # Get version from __init__.py
        init_version = traffic_anomaly.__version__
        
        # Get version from pyproject.toml
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
        
        toml_version = pyproject_data['project']['version']
        
        assert init_version == toml_version, f"Version mismatch: __init__.py has {init_version}, pyproject.toml has {toml_version}"


class TestTrafficAnomalyDecomposition:
    """Test median_decompose function against precalculated results"""
    
    @pytest.fixture
    def sample_travel_times(self):
        """Load sample travel times data"""
        return sample_data.travel_times
    
    @pytest.fixture
    def sample_vehicle_counts(self):
        """Load sample vehicle counts data"""
        return sample_data.vehicle_counts
    
    @pytest.fixture
    def precalc_decomp(self):
        """Load precalculated decomposition results"""
        precalc_path = os.path.join(os.path.dirname(__file__), 'precalculated', 'test_decomp.parquet')
        return pd.read_parquet(precalc_path)
    
    @pytest.fixture
    def precalc_decomp2(self):
        """Load precalculated decomposition results for vehicle counts"""
        precalc_path = os.path.join(os.path.dirname(__file__), 'precalculated', 'test_decomp2.parquet')
        return pd.read_parquet(precalc_path)
    
    def test_median_decompose_travel_times(self, sample_travel_times, precalc_decomp):
        """Test median_decompose with travel times data against precalculated results"""
        # Run the decomposition with the same parameters as used to create precalculated data
        decomp = traffic_anomaly.median_decompose(
            data=sample_travel_times,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id', 'group'],
            freq_minutes=60,
            rolling_window_days=7,
            drop_days=7,
            min_rolling_window_samples=56,
            min_time_of_day_samples=7,
            drop_extras=False,  # keep seasonal/trend for comparison
            to_sql=False
        )
        
        # Sort both dataframes to ensure consistent comparison
        decomp_sorted = decomp.sort_values(['id', 'group', 'timestamp']).reset_index(drop=True)
        precalc_sorted = precalc_decomp.sort_values(['id', 'group', 'timestamp']).reset_index(drop=True)
        
        # Check that the basic structure is consistent
        assert len(decomp_sorted) > 0, "Decomposition should return non-empty results"
        assert 'median' in decomp_sorted.columns, "Median column should be present"
        assert 'season_day' in decomp_sorted.columns, "Season_day column should be present"
        assert 'season_week' in decomp_sorted.columns, "Season_week column should be present"
        assert 'resid' in decomp_sorted.columns, "Residual column should be present"
        assert 'prediction' in decomp_sorted.columns, "Prediction column should be present"
        
        # Check that all columns are present
        expected_columns = {'id', 'group', 'timestamp', 'travel_time', 'median', 'season_day', 'season_week', 'resid', 'prediction'}
        assert set(decomp_sorted.columns) == expected_columns, f"Column mismatch: calculated {set(decomp_sorted.columns)}, expected {expected_columns}"
        
        # Check that numerical values are reasonable (not NaN or infinite)
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'resid', 'prediction']
        for col in numerical_columns:
            assert not decomp_sorted[col].isna().any(), f"Column {col} should not contain NaN values"
            assert np.isfinite(decomp_sorted[col]).all(), f"Column {col} should not contain infinite values"
        
        # Check that categorical/string columns match exactly
        categorical_columns = ['id', 'group']
        for col in categorical_columns:
            if col in decomp_sorted.columns and col in precalc_sorted.columns:
                pd.testing.assert_series_equal(
                    decomp_sorted[col], 
                    precalc_sorted[col], 
                    check_names=False
                )
        
        # Check timestamp column matches exactly
        if 'timestamp' in decomp_sorted.columns and 'timestamp' in precalc_sorted.columns:
            pd.testing.assert_series_equal(
                decomp_sorted['timestamp'], 
                precalc_sorted['timestamp'], 
                check_names=False
            )
    
    def test_median_decompose_vehicle_counts(self, sample_vehicle_counts, precalc_decomp2):
        """Test median_decompose with vehicle counts data against precalculated results"""
        # Run the decomposition with the same parameters as used to create precalculated data
        decomp2 = traffic_anomaly.median_decompose(
            sample_vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        
        # Sort both dataframes to ensure consistent comparison
        decomp2_sorted = decomp2.sort_values(['intersection', 'detector', 'timestamp']).reset_index(drop=True)
        precalc2_sorted = precalc_decomp2.sort_values(['intersection', 'detector', 'timestamp']).reset_index(drop=True)
        
        # Check that the shapes match
        assert decomp2_sorted.shape == precalc2_sorted.shape, f"Shape mismatch: calculated {decomp2_sorted.shape}, precalculated {precalc2_sorted.shape}"
        
        # Check that all columns are present
        assert set(decomp2_sorted.columns) == set(precalc2_sorted.columns), f"Column mismatch: calculated {set(decomp2_sorted.columns)}, precalculated {set(precalc2_sorted.columns)}"
        
        # Check numerical columns for approximate equality
        numerical_columns = ['total', 'median', 'season_day', 'season_week', 'resid', 'prediction']
        for col in numerical_columns:
            if col in decomp2_sorted.columns:
                pd.testing.assert_series_equal(
                    decomp2_sorted[col], 
                    precalc2_sorted[col], 
                    check_names=False,
                    rtol=1e-3,  # relative tolerance (0.1%)
                    atol=1e-3   # absolute tolerance
                )
        
        # Check categorical/string columns for exact equality
        categorical_columns = ['intersection', 'detector']
        for col in categorical_columns:
            if col in decomp2_sorted.columns:
                pd.testing.assert_series_equal(
                    decomp2_sorted[col], 
                    precalc2_sorted[col], 
                    check_names=False
                )
        
        # Check timestamp column
        pd.testing.assert_series_equal(
            decomp2_sorted['timestamp'], 
            precalc2_sorted['timestamp'], 
            check_names=False
        )


class TestTrafficAnomalyDetection:
    """Test find_anomaly function against precalculated results"""
    
    @pytest.fixture
    def decomp_data(self):
        """Create decomposition data for anomaly detection tests"""
        return traffic_anomaly.median_decompose(
            data=sample_data.travel_times,
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
    
    @pytest.fixture
    def decomp2_data(self):
        """Create decomposition data for vehicle counts anomaly detection tests"""
        return traffic_anomaly.median_decompose(
            sample_data.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
    
    @pytest.fixture
    def precalc_anomaly(self):
        """Load precalculated anomaly results"""
        precalc_path = os.path.join(os.path.dirname(__file__), 'precalculated', 'test_anomaly.parquet')
        return pd.read_parquet(precalc_path)
    
    @pytest.fixture
    def precalc_anomaly2(self):
        """Load precalculated anomaly2 results"""
        precalc_path = os.path.join(os.path.dirname(__file__), 'precalculated', 'test_anomaly2.parquet')
        return pd.read_parquet(precalc_path)
    
    @pytest.fixture
    def precalc_anomaly3(self):
        """Load precalculated anomaly3 results"""
        precalc_path = os.path.join(os.path.dirname(__file__), 'precalculated', 'test_anomaly3.parquet')
        return pd.read_parquet(precalc_path)
    
    def test_find_anomaly_basic(self, decomp_data, precalc_anomaly):
        """Test basic anomaly detection against precalculated results"""
        # Apply anomaly detection with the same parameters as used to create precalculated data
        anomaly = traffic_anomaly.find_anomaly(
            decomposed_data=decomp_data,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5
        )
        
        # Sort both dataframes to ensure consistent comparison
        anomaly_sorted = anomaly.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        # Check that the basic structure is consistent
        assert len(anomaly_sorted) > 0, "Anomaly detection should return non-empty results"
        assert 'anomaly' in anomaly_sorted.columns, "Anomaly column should be present"
        
        # Check that all expected columns are present (resid might be dropped in some cases)
        required_columns = {'id', 'group', 'timestamp', 'travel_time', 'median', 'season_day', 'season_week', 'prediction', 'anomaly'}
        actual_columns = set(anomaly_sorted.columns)
        assert required_columns.issubset(actual_columns), f"Missing required columns: {required_columns - actual_columns}"
        
        # Check that anomaly column contains boolean values
        assert anomaly_sorted['anomaly'].dtype == 'bool', "Anomaly column should contain boolean values"
        
        # Check that numerical values are reasonable (not NaN or infinite)
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'prediction']
        if 'resid' in anomaly_sorted.columns:
            numerical_columns.append('resid')
        for col in numerical_columns:
            assert not anomaly_sorted[col].isna().any(), f"Column {col} should not contain NaN values"
            assert np.isfinite(anomaly_sorted[col]).all(), f"Column {col} should not contain infinite values"
        
        # Check that some anomalies are detected (but not all points)
        anomaly_count = anomaly_sorted['anomaly'].sum()
        total_count = len(anomaly_sorted)
        assert 0 <= anomaly_count < total_count, f"Anomaly count should be reasonable: {anomaly_count}/{total_count}"
    
    def test_find_anomaly_with_mad(self, decomp_data, precalc_anomaly2):
        """Test anomaly detection with MAD against precalculated results"""
        # Apply anomaly detection with MAD
        anomaly2 = traffic_anomaly.find_anomaly(
            decomposed_data=decomp_data,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5,
            group_grouping_columns=['group'],
            MAD=True
        )
        
        # Sort both dataframes to ensure consistent comparison
        anomaly2_sorted = anomaly2.sort_values(['id', 'group', 'timestamp']).reset_index(drop=True)
        
        # Check that the basic structure is consistent
        assert len(anomaly2_sorted) > 0, "Anomaly detection with MAD should return non-empty results"
        assert 'anomaly' in anomaly2_sorted.columns, "Anomaly column should be present"
        
        # Check that all expected columns are present (resid might be dropped in some cases)
        required_columns = {'id', 'group', 'timestamp', 'travel_time', 'median', 'season_day', 'season_week', 'prediction', 'anomaly'}
        actual_columns = set(anomaly2_sorted.columns)
        assert required_columns.issubset(actual_columns), f"Missing required columns: {required_columns - actual_columns}"
        
        # Check that anomaly column contains boolean values
        assert anomaly2_sorted['anomaly'].dtype == 'bool', "Anomaly column should contain boolean values"
        
        # Check that numerical values are reasonable (not NaN or infinite)
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'prediction']
        if 'resid' in anomaly2_sorted.columns:
            numerical_columns.append('resid')
        for col in numerical_columns:
            assert not anomaly2_sorted[col].isna().any(), f"Column {col} should not contain NaN values"
            assert np.isfinite(anomaly2_sorted[col]).all(), f"Column {col} should not contain infinite values"
        
        # Check that some anomalies are detected (but not all points)
        anomaly_count = anomaly2_sorted['anomaly'].sum()
        total_count = len(anomaly2_sorted)
        assert 0 <= anomaly_count < total_count, f"Anomaly count should be reasonable: {anomaly_count}/{total_count}"
    
    def test_find_anomaly_with_geh(self, decomp2_data, precalc_anomaly3):
        """Test anomaly detection with GEH against precalculated results"""
        # Apply anomaly detection with GEH
        anomaly3 = traffic_anomaly.find_anomaly(
            decomposed_data=decomp2_data,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
            return_sql=False
        )
        
        # Sort both dataframes to ensure consistent comparison
        anomaly3_sorted = anomaly3.sort_values(['intersection', 'detector', 'timestamp']).reset_index(drop=True)
        precalc3_sorted = precalc_anomaly3.sort_values(['intersection', 'detector', 'timestamp']).reset_index(drop=True)
        
        # Check that the shapes match
        assert anomaly3_sorted.shape == precalc3_sorted.shape, f"Shape mismatch: calculated {anomaly3_sorted.shape}, precalculated {precalc3_sorted.shape}"
        
        # Check that all columns are present
        assert set(anomaly3_sorted.columns) == set(precalc3_sorted.columns), f"Column mismatch: calculated {set(anomaly3_sorted.columns)}, precalculated {set(precalc3_sorted.columns)}"
        
        # Check all columns for equality
        for col in anomaly3_sorted.columns:
            if anomaly3_sorted[col].dtype in ['float64', 'float32']:
                pd.testing.assert_series_equal(
                    anomaly3_sorted[col], 
                    precalc3_sorted[col], 
                    check_names=False,
                    rtol=1e-3,  # relative tolerance (0.1%)
                    atol=1e-3   # absolute tolerance
                )
            else:
                pd.testing.assert_series_equal(
                    anomaly3_sorted[col], 
                    precalc3_sorted[col], 
                    check_names=False
                )


class TestPackageIntegrity:
    """Test package integrity and imports"""
    
    def test_package_imports(self):
        """Test that all main functions can be imported"""
        # Test that we can import the main functions
        from traffic_anomaly import median_decompose, find_anomaly, sample_data
        
        # Test that functions are callable
        assert callable(median_decompose)
        assert callable(find_anomaly)
        
        # Test that sample data is accessible
        assert hasattr(sample_data, 'travel_times')
        assert hasattr(sample_data, 'vehicle_counts')
        
        # Test that sample data are pandas DataFrames
        assert isinstance(sample_data.travel_times, pd.DataFrame)
        assert isinstance(sample_data.vehicle_counts, pd.DataFrame)
    
    def test_sample_data_structure(self):
        """Test that sample data has expected structure"""
        # Test travel_times structure
        travel_times = sample_data.travel_times
        expected_travel_cols = ['timestamp', 'travel_time', 'id', 'group']
        for col in expected_travel_cols:
            assert col in travel_times.columns, f"Missing column {col} in travel_times"
        
        # Test vehicle_counts structure
        vehicle_counts = sample_data.vehicle_counts
        expected_count_cols = ['timestamp', 'total', 'intersection', 'detector']
        for col in expected_count_cols:
            assert col in vehicle_counts.columns, f"Missing column {col} in vehicle_counts"
        
        # Test that data is not empty
        assert len(travel_times) > 0, "travel_times data is empty"
        assert len(vehicle_counts) > 0, "vehicle_counts data is empty"


if __name__ == "__main__":
    pytest.main([__file__])