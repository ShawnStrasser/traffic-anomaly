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
        pyproject_path = project_root / 'pyproject.toml'
        
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
        
        toml_version = pyproject_data['project']['version']
        
        assert init_version == toml_version, f"Version mismatch: __init__.py has {init_version}, pyproject.toml has {toml_version}"


class TestTrafficAnomalyFunctions:
    """Test traffic anomaly detection functions against precalculated results"""
    
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
        return pd.read_parquet('tests/precalculated/test_decomp.parquet')
    
    @pytest.fixture
    def precalc_anomaly(self):
        """Load precalculated anomaly detection results"""
        return pd.read_parquet('tests/precalculated/test_anomaly.parquet')
    
    @pytest.fixture
    def precalc_anomaly2(self):
        """Load precalculated anomaly detection results with MAD"""
        return pd.read_parquet('tests/precalculated/test_anomaly2.parquet')
    
    @pytest.fixture
    def precalc_decomp2(self):
        """Load precalculated decomposition results for vehicle counts"""
        return pd.read_parquet('tests/precalculated/test_decomp2.parquet')
    
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
        
        # Check that the shapes match
        assert decomp_sorted.shape == precalc_sorted.shape, f"Shape mismatch: calculated {decomp_sorted.shape}, precalculated {precalc_sorted.shape}"
        
        # Compare numerical columns with reasonable tolerance
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'resid', 'prediction']
        for col in numerical_columns:
            if col in decomp_sorted.columns and col in precalc_sorted.columns:
                # Use numpy allclose for numerical comparison with reasonable tolerance
                are_close = np.allclose(
                    decomp_sorted[col].values, 
                    precalc_sorted[col].values, 
                    rtol=1e-2,  # 1% relative tolerance
                    atol=1e-2,  # 0.01 absolute tolerance
                    equal_nan=True
                )
                if not are_close:
                    # Calculate differences for debugging
                    diff = np.abs(decomp_sorted[col].values - precalc_sorted[col].values)
                    max_diff = np.nanmax(diff)
                    mean_diff = np.nanmean(diff)
                    print(f"Column {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                    
                assert are_close, f"Column {col} values don't match precalculated results within tolerance"
        
        # Compare categorical/string columns exactly
        categorical_columns = ['id', 'group', 'timestamp']
        for col in categorical_columns:
            if col in decomp_sorted.columns and col in precalc_sorted.columns:
                pd.testing.assert_series_equal(
                    decomp_sorted[col], 
                    precalc_sorted[col], 
                    check_names=False
                )
    
    def test_median_decompose_vehicle_counts(self, sample_vehicle_counts, precalc_decomp2):
        """Test median_decompose with vehicle counts data against precalculated results"""
        # Run the decomposition with the same parameters as used to create precalculated data
        decomp2 = traffic_anomaly.median_decompose(
            sample_vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['id'],
            freq_minutes=60,
            rolling_window_days=7,
            drop_days=7,
            min_rolling_window_samples=56,
            min_time_of_day_samples=7,
            drop_extras=False,
            to_sql=False
        )
        
        # Sort both dataframes to ensure consistent comparison
        decomp2_sorted = decomp2.sort_values(['id', 'timestamp']).reset_index(drop=True)
        precalc2_sorted = precalc_decomp2.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        # Check that the shapes match
        assert decomp2_sorted.shape == precalc2_sorted.shape, f"Shape mismatch: calculated {decomp2_sorted.shape}, precalculated {precalc2_sorted.shape}"
        
        # Compare numerical columns with reasonable tolerance
        numerical_columns = ['total', 'median', 'season_day', 'season_week', 'resid', 'prediction']
        for col in numerical_columns:
            if col in decomp2_sorted.columns and col in precalc2_sorted.columns:
                # Use numpy allclose for numerical comparison with reasonable tolerance
                are_close = np.allclose(
                    decomp2_sorted[col].values, 
                    precalc2_sorted[col].values, 
                    rtol=1e-2,  # 1% relative tolerance
                    atol=1e-2,  # 0.01 absolute tolerance
                    equal_nan=True
                )
                if not are_close:
                    # Calculate differences for debugging
                    diff = np.abs(decomp2_sorted[col].values - precalc2_sorted[col].values)
                    max_diff = np.nanmax(diff)
                    mean_diff = np.nanmean(diff)
                    print(f"Column {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                    
                assert are_close, f"Column {col} values don't match precalculated results within tolerance"
    
    def test_find_anomaly_basic(self, sample_travel_times, precalc_anomaly):
        """Test basic anomaly detection against precalculated results"""
        # First run decomposition to get the input data
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
            drop_extras=False,
            to_sql=False
        )
        
        # Apply anomaly detection with the same parameters as used to create precalculated data
        anomaly = traffic_anomaly.find_anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5
        )
        
        # Sort both dataframes to ensure consistent comparison
        anomaly_sorted = anomaly.sort_values(['id', 'timestamp']).reset_index(drop=True)
        precalc_sorted = precalc_anomaly.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        # Check that the shapes match
        assert anomaly_sorted.shape == precalc_sorted.shape, f"Shape mismatch: calculated {anomaly_sorted.shape}, precalculated {precalc_sorted.shape}"
        
        # Compare numerical columns with reasonable tolerance
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'prediction']
        if 'resid' in anomaly_sorted.columns:
            numerical_columns.append('resid')
            
        for col in numerical_columns:
            if col in anomaly_sorted.columns and col in precalc_sorted.columns:
                # Use numpy allclose for numerical comparison with reasonable tolerance
                are_close = np.allclose(
                    anomaly_sorted[col].values, 
                    precalc_sorted[col].values, 
                    rtol=1e-2,  # 1% relative tolerance
                    atol=1e-2,  # 0.01 absolute tolerance
                    equal_nan=True
                )
                if not are_close:
                    # Calculate differences for debugging
                    diff = np.abs(anomaly_sorted[col].values - precalc_sorted[col].values)
                    max_diff = np.nanmax(diff)
                    mean_diff = np.nanmean(diff)
                    print(f"Column {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                    
                assert are_close, f"Column {col} values don't match precalculated results within tolerance"
        
        # Compare boolean anomaly column exactly
        if 'anomaly' in anomaly_sorted.columns and 'anomaly' in precalc_sorted.columns:
            pd.testing.assert_series_equal(
                anomaly_sorted['anomaly'], 
                precalc_sorted['anomaly'], 
                check_names=False
            )
    
    def test_find_anomaly_with_mad(self, sample_travel_times, precalc_anomaly2):
        """Test anomaly detection with MAD against precalculated results"""
        # First run decomposition to get the input data
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
            drop_extras=False,
            to_sql=False
        )
        
        # Apply anomaly detection with MAD
        anomaly2 = traffic_anomaly.find_anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5,
            group_grouping_columns=['group'],
            MAD=True
        )
        
        # Sort both dataframes to ensure consistent comparison
        anomaly2_sorted = anomaly2.sort_values(['id', 'group', 'timestamp']).reset_index(drop=True)
        precalc2_sorted = precalc_anomaly2.sort_values(['id', 'group', 'timestamp']).reset_index(drop=True)
        
        # Check that the shapes match
        assert anomaly2_sorted.shape == precalc2_sorted.shape, f"Shape mismatch: calculated {anomaly2_sorted.shape}, precalculated {precalc2_sorted.shape}"
        
        # Compare numerical columns with reasonable tolerance
        numerical_columns = ['travel_time', 'median', 'season_day', 'season_week', 'prediction']
        if 'resid' in anomaly2_sorted.columns:
            numerical_columns.append('resid')
            
        for col in numerical_columns:
            if col in anomaly2_sorted.columns and col in precalc2_sorted.columns:
                # Use numpy allclose for numerical comparison with reasonable tolerance
                are_close = np.allclose(
                    anomaly2_sorted[col].values, 
                    precalc2_sorted[col].values, 
                    rtol=1e-2,  # 1% relative tolerance
                    atol=1e-2,  # 0.01 absolute tolerance
                    equal_nan=True
                )
                if not are_close:
                    # Calculate differences for debugging
                    diff = np.abs(anomaly2_sorted[col].values - precalc2_sorted[col].values)
                    max_diff = np.nanmax(diff)
                    mean_diff = np.nanmean(diff)
                    print(f"Column {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                    
                assert are_close, f"Column {col} values don't match precalculated results within tolerance"
        
        # Compare boolean anomaly column exactly
        if 'anomaly' in anomaly2_sorted.columns and 'anomaly' in precalc2_sorted.columns:
            pd.testing.assert_series_equal(
                anomaly2_sorted['anomaly'], 
                precalc2_sorted['anomaly'], 
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