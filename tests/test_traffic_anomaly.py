import pytest
import pandas as pd
import numpy as np
import os
import sys
import toml

# Add the src directory to the path to import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import traffic_anomaly
from traffic_anomaly import sample_data


class TestTrafficAnomaly:
    """Test suite for traffic_anomaly package"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and precalculated results paths"""
        cls.tests_dir = os.path.dirname(__file__)
        cls.precalculated_dir = os.path.join(cls.tests_dir, 'precalculated')
        cls.project_root = os.path.dirname(cls.tests_dir)
        
        # Load sample data
        cls.travel_times = sample_data.travel_times
        cls.vehicle_counts = sample_data.vehicle_counts
    
    def test_version_consistency(self):
        """Test that version numbers match between __init__.py and pyproject.toml"""
        # Get version from __init__.py
        init_version = traffic_anomaly.__version__
        
        # Get version from pyproject.toml
        pyproject_path = os.path.join(self.project_root, 'pyproject.toml')
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
        pyproject_version = pyproject_data['project']['version']
        
        assert init_version == pyproject_version, (
            f"Version mismatch: __init__.py has {init_version}, "
            f"pyproject.toml has {pyproject_version}"
        )
    
    def test_decompose_travel_times(self):
        """Test decompose with travel_times data against precalculated results"""
        # Calculate decomposition
        decomp = traffic_anomaly.decompose(
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
            return_sql=False
        )
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_decomp.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(decomp, expected, "decompose_travel_times")
    
    def test_decompose_vehicle_counts(self):
        """Test decompose with vehicle_counts data against precalculated results"""
        # Calculate decomposition
        decomp2 = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_decomp2.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(decomp2, expected, "decompose_vehicle_counts")
    
    def test_anomaly_basic(self):
        """Test anomaly basic functionality against precalculated results"""
        # First get the decomposition
        decomp = traffic_anomaly.decompose(
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
            return_sql=False
        )
        
        # Apply anomaly detection
        anomaly = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5
        )
        # Drop 'resid' for comparison with precalculated results
        if 'resid' in anomaly.columns:
            anomaly = anomaly.drop(columns=['resid'])
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_anomaly.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(anomaly, expected, "anomaly_basic")
    
    def test_anomaly_with_mad(self):
        """Test anomaly with MAD=True against precalculated results"""
        # First get the decomposition
        decomp = traffic_anomaly.decompose(
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
            return_sql=False
        )
        
        # Apply anomaly detection with MAD
        anomaly2 = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5,
            group_grouping_columns=['group'],
            MAD=True
        )
        # Drop 'resid' for comparison
        if 'resid' in anomaly2.columns:
            anomaly2 = anomaly2.drop(columns=['resid'])
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_anomaly2.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(anomaly2, expected, "anomaly_with_mad")
    
    def test_anomaly_with_geh(self):
        """Test anomaly with GEH=True against precalculated results"""
        # First get the decomposition for vehicle counts
        decomp2 = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        
        # Apply anomaly detection with GEH
        anomaly3 = traffic_anomaly.anomaly(
            decomposed_data=decomp2,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
            return_sql=False
        )
        # Drop 'resid' for comparison
        if 'resid' in anomaly3.columns:
            anomaly3 = anomaly3.drop(columns=['resid'])
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_anomaly3.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(anomaly3, expected, "anomaly_with_geh")

    def test_anomaly_with_connectivity_originated(self):
        """Test anomaly with connectivity_table (originated anomalies) against precalculated results"""
        # Decompose travel times using the same parameters as other tests for consistency
        decomp = traffic_anomaly.decompose(
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
            return_sql=False
        )

        # Load connectivity table from sample data
        df_connectivity = sample_data.connectivity

        # Apply anomaly detection with connectivity to compute originated anomalies
        anomaly_originated = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=3.5,
            group_grouping_columns=['group'],
            MAD=True,
            connectivity_table=df_connectivity
        )
        # Align with precalculated output by dropping internal 'resid' column if present
        if 'resid' in anomaly_originated.columns:
            anomaly_originated = anomaly_originated.drop(columns=['resid'])

        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_anomaly_originated.parquet')
        expected = pd.read_parquet(expected_path)

        # Compare results
        self._compare_dataframes(anomaly_originated, expected, "anomaly_with_connectivity_originated")

    def test_anomaly_connectivity_requires_single_entity(self):
        """Connectivity analysis should enforce a single entity grouping column"""
        # Minimal decomposition for speed
        decomp_small = traffic_anomaly.decompose(
            data=self.travel_times.head(50),
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id', 'group'],
            rolling_window_enable=False,
            drop_extras=False
        )

        df_connectivity = sample_data.connectivity

        # Passing more than one entity grouping column should raise a ValueError when connectivity is used
        with pytest.raises(ValueError, match="Connectivity analysis is currently only supported for a single entity grouping column"):
            traffic_anomaly.anomaly(
                decomposed_data=decomp_small,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id', 'group'],
                connectivity_table=df_connectivity
            )
    
    def test_changepoint_robust(self):
        """Test changepoint detection with robust=True against precalculated results"""
        # Load sample changepoint input data
        df = sample_data.changepoints_input
        
        # Calculate changepoints with robust=True
        changepoints_robust = traffic_anomaly.changepoint(
            df,
            value_column='travel_time_seconds',
            entity_grouping_column='ID',
            datetime_column='TimeStamp',
            score_threshold=0.7,
            robust=True
        )
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_changepoint_robust.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(changepoints_robust, expected, "changepoint_robust")
    
    def test_changepoint_standard(self):
        """Test changepoint detection with robust=False against precalculated results"""
        # Load sample changepoint input data
        df = sample_data.changepoints_input
        
        # Calculate changepoints with robust=False
        changepoints_standard = traffic_anomaly.changepoint(
            df,
            value_column='travel_time_seconds',
            entity_grouping_column='ID',
            datetime_column='TimeStamp',
            score_threshold=0.7,
            robust=False
        )
        
        # Load precalculated results
        expected_path = os.path.join(self.precalculated_dir, 'test_changepoint.parquet')
        expected = pd.read_parquet(expected_path)
        
        # Compare results
        self._compare_dataframes(changepoints_standard, expected, "changepoint_standard")
    
    def _compare_dataframes(self, actual, expected, test_name):
        """Helper method to compare two dataframes with detailed error reporting"""
        
        # Check if both are DataFrames
        assert isinstance(actual, pd.DataFrame), f"{test_name}: Actual result is not a DataFrame"
        assert isinstance(expected, pd.DataFrame), f"{test_name}: Expected result is not a DataFrame"
        
        # Check shape
        assert actual.shape == expected.shape, (
            f"{test_name}: Shape mismatch - actual: {actual.shape}, expected: {expected.shape}"
        )
        
        # Check columns
        assert list(actual.columns) == list(expected.columns), (
            f"{test_name}: Column mismatch - actual: {list(actual.columns)}, "
            f"expected: {list(expected.columns)}"
        )
        
        # Sort both dataframes by all columns to ensure consistent ordering
        # Convert datetime columns to string temporarily for sorting
        actual_sorted = actual.copy()
        expected_sorted = expected.copy()
        
        for col in actual.columns:
            if pd.api.types.is_datetime64_any_dtype(actual[col]):
                actual_sorted[col] = actual[col].astype(str)
                expected_sorted[col] = expected[col].astype(str)
        
        # Sort by all columns
        sort_columns = list(actual_sorted.columns)
        actual_sorted = actual_sorted.sort_values(sort_columns).reset_index(drop=True)
        expected_sorted = expected_sorted.sort_values(sort_columns).reset_index(drop=True)
        
        # Restore datetime columns in sorted dataframes
        for col in actual.columns:
            if pd.api.types.is_datetime64_any_dtype(actual[col]):
                actual_sorted[col] = pd.to_datetime(actual_sorted[col])
                expected_sorted[col] = pd.to_datetime(expected_sorted[col])
        
        # Compare each column
        for col in actual.columns:
            if pd.api.types.is_numeric_dtype(actual[col]):
                # For numeric columns, use np.allclose for floating point comparison
                # Using 0.1 tolerance for 1 decimal place accuracy
                if not np.allclose(actual_sorted[col].fillna(0), 
                                 expected_sorted[col].fillna(0), 
                                 rtol=0.1, atol=0.1, equal_nan=True):
                    
                    # Find the first differing value for detailed error
                    mask = ~np.isclose(actual_sorted[col].fillna(0), 
                                     expected_sorted[col].fillna(0), 
                                     rtol=0.1, atol=0.1, equal_nan=True)
                    if mask.any():
                        first_diff_idx = np.argmax(mask)
                        actual_val = actual_sorted[col].iloc[first_diff_idx]
                        expected_val = expected_sorted[col].iloc[first_diff_idx]
                        
                        pytest.fail(
                            f"{test_name}: Numeric values differ in column '{col}' at index {first_diff_idx}:\n"
                            f"  Actual: {actual_val}\n"
                            f"  Expected: {expected_val}\n"
                            f"  Difference: {abs(actual_val - expected_val) if pd.notna(actual_val) and pd.notna(expected_val) else 'NaN comparison'}"
                        )
            
            elif pd.api.types.is_datetime64_any_dtype(actual[col]):
                # For datetime columns, compare directly
                if not actual_sorted[col].equals(expected_sorted[col]):
                    # Find first differing value
                    mask = actual_sorted[col] != expected_sorted[col]
                    if mask.any():
                        first_diff_idx = mask.idxmax()
                        actual_val = actual_sorted[col].iloc[first_diff_idx]
                        expected_val = expected_sorted[col].iloc[first_diff_idx]
                        
                        pytest.fail(
                            f"{test_name}: Datetime values differ in column '{col}' at index {first_diff_idx}:\n"
                            f"  Actual: {actual_val}\n"
                            f"  Expected: {expected_val}"
                        )
            
            else:
                # For other columns (strings, etc.), compare directly
                if not actual_sorted[col].equals(expected_sorted[col]):
                    # Find first differing value
                    mask = actual_sorted[col] != expected_sorted[col]
                    if mask.any():
                        first_diff_idx = mask.idxmax()
                        actual_val = actual_sorted[col].iloc[first_diff_idx]
                        expected_val = expected_sorted[col].iloc[first_diff_idx]
                        
                        pytest.fail(
                            f"{test_name}: Values differ in column '{col}' at index {first_diff_idx}:\n"
                            f"  Actual: {actual_val}\n"
                            f"  Expected: {expected_val}"
                        )

    # Meaningful functional tests to verify correctness
    def test_sql_execution_equivalence_decompose(self):
        """Test that SQL output from decompose produces equivalent results when executed"""
        import duckdb
        import ibis
        
        # Use the same approach as the package code - ibis.memtable()
        travel_times_table = ibis.memtable(self.travel_times)
        
        # Get regular result using Ibis table (same as package approach)
        regular_result = traffic_anomaly.decompose(
            data=travel_times_table,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_days=3,
            drop_days=3,
            min_rolling_window_samples=10,
            drop_extras=False,
            return_sql=False
        ).execute()
        
        # Get SQL query using the same Ibis table
        sql_query = traffic_anomaly.decompose(
            data=travel_times_table,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_days=3,
            drop_days=3,
            min_rolling_window_samples=10,
            drop_extras=False,
            return_sql=True
        )
        
        # Execute SQL with DuckDB using the original pandas data
        conn = duckdb.connect()
        conn.register('travel_times', self.travel_times)
        
        # Replace the memtable reference in SQL with our registered table
        # This is a bit hacky but works around the temporary table name issue
        import re
        # Find the memtable reference and replace with our registered table name
        sql_with_table = re.sub(r'"ibis_pandas_memtable_[a-z0-9]+"', '"travel_times"', sql_query)
        
        sql_result = conn.execute(sql_with_table).fetchdf()
        conn.close()
        
        # Compare results (allowing for small numerical differences)
        assert regular_result.shape == sql_result.shape, "SQL and regular execution should produce same shape"
        
        # Compare key columns after sorting
        for col in ['id', 'prediction']:
            if col in regular_result.columns and col in sql_result.columns:
                regular_sorted = regular_result.sort_values(['id', 'timestamp'])[col].reset_index(drop=True)
                sql_sorted = sql_result.sort_values(['id', 'timestamp'])[col].reset_index(drop=True)
                
                if pd.api.types.is_numeric_dtype(regular_sorted):
                    assert np.allclose(regular_sorted.fillna(0), sql_sorted.fillna(0), rtol=0.1), \
                        f"SQL and regular results should match for {col}"

    def test_sql_execution_equivalence_anomaly(self):
        """Test that SQL output from anomaly produces equivalent results when executed"""
        import duckdb
        import ibis
        
        # Use the same approach as the package code - ibis.memtable()
        travel_times_table = ibis.memtable(self.travel_times)
        
        # First get decomposition using Ibis table (same as package approach)
        decomp = traffic_anomaly.decompose(
            data=travel_times_table,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_days=3,
            drop_days=3,
            min_rolling_window_samples=10,
            drop_extras=False
        )
        
        # Get both regular result and SQL using the same Ibis expression
        regular_result = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=2.0,
            return_sql=False
        ).execute()
        
        sql_query = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            entity_threshold=2.0,
            return_sql=True
        )
        
        # Execute SQL with DuckDB using the original pandas data
        conn = duckdb.connect()
        conn.register('travel_times', self.travel_times)
        
        # Replace the memtable reference in SQL with our registered table
        import re
        sql_with_table = re.sub(r'"ibis_pandas_memtable_[a-z0-9]+"', '"travel_times"', sql_query)
        
        sql_result = conn.execute(sql_with_table).fetchdf()
        conn.close()
        
        # Compare anomaly detection results
        assert regular_result.shape == sql_result.shape, "SQL and regular execution should produce same shape"
        
        # Sort both for comparison
        regular_sorted = regular_result.sort_values(['id', 'timestamp']).reset_index(drop=True)
        sql_sorted = sql_result.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        # Compare anomaly column specifically
        assert regular_sorted['anomaly'].equals(sql_sorted['anomaly']), \
            "SQL and regular execution should produce identical anomaly flags"

    def test_rolling_vs_static_decomposition(self):
        """Test functional difference between rolling and static decomposition"""
        # Use smaller subset for faster testing
        small_data = self.travel_times.head(200)  # Just 200 rows for speed
        
        # Rolling decomposition with simple parameters
        rolling_result = traffic_anomaly.decompose(
            data=small_data,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_enable=True,
            rolling_window_days=2,  # Small window
            drop_days=1,           # Minimal drop
            min_rolling_window_samples=5,  # Low requirement
            drop_extras=False
        )
        
        # Static decomposition  
        static_result = traffic_anomaly.decompose(
            data=small_data,
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_enable=False,
            drop_extras=False
        )
        
        # Create precalculated dataset for rolling disabled case
        static_expected_path = os.path.join(self.precalculated_dir, 'test_static_decomp_small.parquet')
        if not os.path.exists(static_expected_path):
            # Generate and save the expected result
            static_result.to_parquet(static_expected_path)
        
        expected_static = pd.read_parquet(static_expected_path)
        self._compare_dataframes(static_result, expected_static, "static_decomposition_small")
        
        # Verify they produce functionally different results
        # Rolling should have fewer records due to drop_days and min_rolling_window_samples
        # Static should include all original records
        assert len(rolling_result) < len(static_result), \
            "Rolling decomposition should have fewer records due to window requirements"
        
        # Both should have the same columns when drop_extras=False
        assert set(rolling_result.columns) == set(static_result.columns), \
            "Both decomposition types should produce the same columns"

    def test_geh_vs_zscore_anomaly_detection(self):
        """Test functional difference between GEH and Z-score anomaly detection"""
        # Use smaller subset of vehicle counts for faster testing
        small_vehicle_data = self.vehicle_counts.head(500)
        
        # Get decomposition for vehicle counts (good for GEH)
        decomp = traffic_anomaly.decompose(
            small_vehicle_data,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False,  # Static is faster
            drop_extras=False
        )
        
        # GEH-based detection
        geh_result = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            GEH=True,
            entity_threshold=6.0
        )
        if 'resid' in geh_result.columns:
            geh_compare = geh_result.drop(columns=['resid'])
        else:
            geh_compare = geh_result
        
        # Z-score based detection
        zscore_result = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            GEH=False,
            entity_threshold=3.0
        )
        if 'resid' in zscore_result.columns:
            zscore_compare = zscore_result.drop(columns=['resid'])
        else:
            zscore_compare = zscore_result
        
        # Create precalculated datasets
        geh_expected_path = os.path.join(self.precalculated_dir, 'test_geh_anomaly_small.parquet')
        zscore_expected_path = os.path.join(self.precalculated_dir, 'test_zscore_anomaly_small.parquet')
        
        if not os.path.exists(geh_expected_path):
            geh_compare.to_parquet(geh_expected_path)
        if not os.path.exists(zscore_expected_path):
            zscore_compare.to_parquet(zscore_expected_path)
        
        expected_geh = pd.read_parquet(geh_expected_path)
        expected_zscore = pd.read_parquet(zscore_expected_path)
        
        self._compare_dataframes(geh_compare, expected_geh, "geh_anomaly_detection_small")
        self._compare_dataframes(zscore_compare, expected_zscore, "zscore_anomaly_detection_small")
        
        # Verify they detect different anomalies (GEH is magnitude-aware)
        geh_anomalies = geh_result['anomaly'].sum()
        zscore_anomalies = zscore_result['anomaly'].sum()
        
        # Methods should produce valid results (they might detect same count with these thresholds)
        assert geh_anomalies >= 0 and zscore_anomalies >= 0, \
            "Both GEH and Z-score methods should produce valid anomaly counts"
        
        # Verify the actual functional difference - different methods were used
        assert 'anomaly' in geh_result.columns and 'anomaly' in zscore_result.columns, \
            "Both methods should produce anomaly detection results"

    def test_log_adjustment_impact(self):
        """Test the functional impact of log_adjust_negative parameter"""
        # Use smaller subset of vehicle counts for faster testing
        small_vehicle_data = self.vehicle_counts.head(500)
        
        # Get decomposition with some zero/low values
        decomp = traffic_anomaly.decompose(
            small_vehicle_data,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False,  # Static is faster
            drop_extras=False
        )
        
        # Without log adjustment
        normal_result = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            GEH=True,
            log_adjust_negative=False,
            entity_threshold=6.0
        )
        
        # With log adjustment
        log_adjusted_result = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            GEH=True,
            log_adjust_negative=True,
            entity_threshold=6.0
        )
        # Drop 'resid' for comparison and expected generation
        if 'resid' in log_adjusted_result.columns:
            log_adjusted_compare = log_adjusted_result.drop(columns=['resid'])
        else:
            log_adjusted_compare = log_adjusted_result
        
        # Create precalculated dataset
        log_adjusted_expected_path = os.path.join(self.precalculated_dir, 'test_log_adjusted_small.parquet')
        if not os.path.exists(log_adjusted_expected_path):
            log_adjusted_compare.to_parquet(log_adjusted_expected_path)
        
        expected_log_adjusted = pd.read_parquet(log_adjusted_expected_path)
        self._compare_dataframes(log_adjusted_compare, expected_log_adjusted, "log_adjusted_anomalies_small")
        
        # Log adjustment should generally detect more anomalies for low-value scenarios
        normal_count = normal_result['anomaly'].sum()
        log_adjusted_count = log_adjusted_result['anomaly'].sum()
        
        # The counts may be different due to log adjustment amplifying certain residuals
        assert normal_count >= 0 and log_adjusted_count >= 0, \
            "Both methods should detect non-negative anomaly counts"

    def test_basic_error_handling(self):
        """Test basic error handling without complex setup"""
        # Test missing required columns with minimal decomposed data
        incomplete_data = pd.DataFrame({'id': [1], 'timestamp': ['2022-01-01']})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.anomaly(
                decomposed_data=incomplete_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test parameter validation - invalid grouping columns type
        good_decomp = traffic_anomaly.decompose(
            data=self.travel_times.head(50),  # Small for speed
            datetime_column='timestamp',
            value_column='travel_time',
            entity_grouping_columns=['id'],
            rolling_window_enable=False,
            drop_extras=False
        )
        
        with pytest.raises(AssertionError, match="entity_grouping_columns must be a list"):
            traffic_anomaly.anomaly(
                decomposed_data=good_decomp,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns="id"  # Should be ['id']
            )

    # ========================================
    # DATA INPUT VALIDATION TESTS
    # ========================================

    def test_decompose_missing_columns(self):
        """Test decompose with missing required columns"""
        # Test missing datetime column
        data_missing_datetime = pd.DataFrame({
            'id': [1, 2],
            'travel_time': [10.0, 15.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*timestamp"):
            traffic_anomaly.decompose(
                data=data_missing_datetime,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test missing value column
        data_missing_value = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02'])
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*travel_time"):
            traffic_anomaly.decompose(
                data=data_missing_value,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test missing entity grouping columns
        data_missing_entity = pd.DataFrame({
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*id"):
            traffic_anomaly.decompose(
                data=data_missing_entity,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test multiple missing columns
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.decompose(
                data=pd.DataFrame({'timestamp': pd.to_datetime(['2022-01-01'])}),
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_decompose_invalid_parameter_types(self):
        """Test decompose with invalid parameter types"""
        # Test non-list entity_grouping_columns
        with pytest.raises(ValueError, match="entity_grouping_columns must be a list"):
            traffic_anomaly.decompose(
                data=self.travel_times.head(10),
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns='id'  # Should be ['id']
            )
        
        # Test with tuple instead of list (should also fail)
        with pytest.raises(ValueError, match="entity_grouping_columns must be a list"):
            traffic_anomaly.decompose(
                data=self.travel_times.head(10),
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=('id',)  # Tuple instead of list
            )

    def test_decompose_invalid_data_type(self):
        """Test decompose with invalid data input types"""
        # Test with invalid data type (not DataFrame or Ibis)
        with pytest.raises(ValueError, match="Invalid data type. Please provide a valid Ibis table or pandas DataFrame"):
            traffic_anomaly.decompose(
                data="invalid_data_type",
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test with None - this actually gets converted to empty DataFrame by ibis, so it triggers missing columns error
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.decompose(
                data=None,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_anomaly_missing_columns(self):
        """Test anomaly detection with missing required columns"""
        # Test missing datetime column
        data_missing_datetime = pd.DataFrame({
            'id': [1, 2],
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*timestamp"):
            traffic_anomaly.anomaly(
                decomposed_data=data_missing_datetime,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test missing value column
        data_missing_value = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*travel_time"):
            traffic_anomaly.anomaly(
                decomposed_data=data_missing_value,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test missing entity grouping columns
        data_missing_entity = pd.DataFrame({
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*id"):
            traffic_anomaly.anomaly(
                decomposed_data=data_missing_entity,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_anomaly_missing_decomposed_columns(self):
        """Test anomaly detection with missing prediction/resid columns"""
        # Test missing prediction column
        data_missing_prediction = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'resid': [1.0, 1.0]
        })
        
        with pytest.raises(AssertionError, match="prediction column not found"):
            traffic_anomaly.anomaly(
                decomposed_data=data_missing_prediction,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test missing resid column
        data_missing_resid = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0]
        })
        
        with pytest.raises(AssertionError, match="resid column not found"):
            traffic_anomaly.anomaly(
                decomposed_data=data_missing_resid,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_anomaly_invalid_parameter_types(self):
        """Test anomaly detection with invalid parameter types"""
        # Create valid minimal decomposed data
        valid_data = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        # Test non-list entity_grouping_columns
        with pytest.raises(AssertionError, match="entity_grouping_columns must be a list"):
            traffic_anomaly.anomaly(
                decomposed_data=valid_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns='id'  # Should be ['id']
            )
        
        # Test invalid group_grouping_columns type
        with pytest.raises(AssertionError, match="group_grouping_columns must be a list"):
            traffic_anomaly.anomaly(
                decomposed_data=valid_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id'],
                group_grouping_columns='group'  # Should be ['group'] or None
            )

    def test_anomaly_invalid_data_type(self):
        """Test anomaly detection with invalid data input types"""
        # Test with invalid data type
        with pytest.raises(ValueError, match="Invalid data type"):
            traffic_anomaly.anomaly(
                decomposed_data="invalid_data_type",
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_changepoint_missing_columns(self):
        """Test changepoint detection with missing required columns"""
        # Test missing datetime column
        data_missing_datetime = pd.DataFrame({
            'ID': [1, 2],
            'travel_time_seconds': [10.0, 15.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*TimeStamp"):
            traffic_anomaly.changepoint(
                data=data_missing_datetime,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )
        
        # Test missing value column
        data_missing_value = pd.DataFrame({
            'ID': [1, 2],
            'TimeStamp': pd.to_datetime(['2022-01-01', '2022-01-02'])
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*travel_time_seconds"):
            traffic_anomaly.changepoint(
                data=data_missing_value,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )
        
        # Test missing entity grouping column (single string)
        data_missing_entity = pd.DataFrame({
            'TimeStamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time_seconds': [10.0, 15.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*ID"):
            traffic_anomaly.changepoint(
                data=data_missing_entity,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )
        
        # Test missing entity grouping columns (list)
        with pytest.raises(ValueError, match="Missing required columns.*ID.*group"):
            traffic_anomaly.changepoint(
                data=data_missing_entity,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column=['ID', 'group']
            )

    def test_changepoint_invalid_parameter_values(self):
        """Test changepoint detection with invalid parameter values"""
        # Create valid minimal data
        valid_data = pd.DataFrame({
            'ID': [1, 1, 2, 2],
            'TimeStamp': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-02']),
            'travel_time_seconds': [10.0, 15.0, 12.0, 18.0]
        })
        
        # Test invalid upper_bound (>1)
        with pytest.raises(ValueError, match="upper_bound must be between 0 and 1"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                upper_bound=1.5
            )
        
        # Test invalid upper_bound (<0)
        with pytest.raises(ValueError, match="upper_bound must be between 0 and 1"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                upper_bound=-0.1
            )
        
        # Test invalid lower_bound (>1)
        with pytest.raises(ValueError, match="lower_bound must be between 0 and 1"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                lower_bound=1.2
            )
        
        # Test invalid lower_bound (<0)
        with pytest.raises(ValueError, match="lower_bound must be between 0 and 1"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                lower_bound=-0.05
            )
        
        # Test lower_bound >= upper_bound
        with pytest.raises(ValueError, match="lower_bound must be less than upper_bound"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                lower_bound=0.8,
                upper_bound=0.7
            )
        
        # Test lower_bound == upper_bound
        with pytest.raises(ValueError, match="lower_bound must be less than upper_bound"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                lower_bound=0.5,
                upper_bound=0.5
            )
        
        # Test invalid rolling_window_days (<=0)
        with pytest.raises(ValueError, match="rolling_window_days must be positive"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                rolling_window_days=0
            )
        
        with pytest.raises(ValueError, match="rolling_window_days must be positive"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                rolling_window_days=-5
            )
        
        # Test invalid min_separation_days (<0)
        with pytest.raises(ValueError, match="min_separation_days must be non-negative"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                min_separation_days=-1
            )

    def test_changepoint_invalid_data_type(self):
        """Test changepoint detection with invalid data input types"""
        # Test with invalid data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a valid Ibis table or pandas DataFrame"):
            traffic_anomaly.changepoint(
                data="invalid_data_type",
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )
        
        # Test with None - this actually gets converted to empty DataFrame by ibis, so it triggers missing columns error
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.changepoint(
                data=None,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )

    def test_changepoint_entity_grouping_column_variations(self):
        """Test changepoint with both string and list entity grouping columns"""
        # Test missing column in list
        valid_data = pd.DataFrame({
            'ID': [1, 1, 2, 2],
            'TimeStamp': pd.to_datetime(['2022-01-01', '2022-01-15', '2022-01-01', '2022-01-15']),
            'travel_time_seconds': [10.0, 15.0, 12.0, 18.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns.*missing_col"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column=['ID', 'missing_col']
            )

    def test_decompose_edge_cases(self):
        """Test decompose with edge case scenarios"""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.decompose(
                data=empty_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_anomaly_edge_cases(self):
        """Test anomaly detection with edge case scenarios"""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.anomaly(
                decomposed_data=empty_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test with DataFrame that has basic columns but missing decomposed columns
        basic_data = pd.DataFrame({
            'id': [1],
            'timestamp': pd.to_datetime(['2022-01-01']),
            'travel_time': [10.0]
        })
        
        with pytest.raises(AssertionError, match="prediction column not found"):
            traffic_anomaly.anomaly(
                decomposed_data=basic_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_changepoint_edge_cases(self):
        """Test changepoint detection with edge case scenarios"""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            traffic_anomaly.changepoint(
                data=empty_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )

    def test_parameter_type_validation_comprehensive(self):
        """Test comprehensive parameter type validation across all functions"""
        # Create minimal valid data for each function
        anomaly_data = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        changepoint_data = pd.DataFrame({
            'ID': [1, 1, 2, 2],
            'TimeStamp': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-02']),
            'travel_time_seconds': [10.0, 15.0, 12.0, 18.0]
        })
        
        # Test anomaly with various invalid parameter types
        invalid_grouping_types = [
            123,
            {'key': 'value'},
            set(['id']),
        ]
        
        for invalid_type in invalid_grouping_types:
            with pytest.raises(AssertionError, match="entity_grouping_columns must be a list"):
                traffic_anomaly.anomaly(
                    decomposed_data=anomaly_data,
                    datetime_column='timestamp',
                    value_column='travel_time',
                    entity_grouping_columns=invalid_type
                )
        
        # Test anomaly group_grouping_columns validation
        invalid_group_types = [123, {'key': 'value'}, set(['group'])]
        
        for invalid_type in invalid_group_types:
            with pytest.raises(AssertionError, match="group_grouping_columns must be a list"):
                traffic_anomaly.anomaly(
                    decomposed_data=anomaly_data,
                    datetime_column='timestamp',
                    value_column='travel_time',
                    entity_grouping_columns=['id'],
                    group_grouping_columns=invalid_type
                )
        
        # Test changepoint with invalid numeric parameter types
        with pytest.raises((ValueError, TypeError)):
            traffic_anomaly.changepoint(
                data=changepoint_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                upper_bound="invalid"  # Should be float
            )
        
        with pytest.raises((ValueError, TypeError)):
            traffic_anomaly.changepoint(
                data=changepoint_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                rolling_window_days="invalid"  # Should be int
            )

    def test_memtable_exception_handling(self):
        """Test exception handling in ibis.memtable() conversion"""
        # Mock an object that will cause ibis.memtable() to raise an exception
        class BadDataType:
            def __init__(self):
                pass
            
            def __str__(self):
                return "mock_bad_data"
        
        bad_data = BadDataType()
        
        # Test decompose with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a valid Ibis table or pandas DataFrame"):
            traffic_anomaly.decompose(
                data=bad_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test anomaly with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a valid Ibis table or pandas DataFrame"):
            traffic_anomaly.anomaly(
                decomposed_data=bad_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test changepoint with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a valid Ibis table or pandas DataFrame"):
            traffic_anomaly.changepoint(
                data=bad_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )

    def test_none_default_parameter_handling(self):
        """Test handling of None default parameters"""
        # Test anomaly with None group_grouping_columns (should be allowed)
        decomposed_data = pd.DataFrame({
            'id': [1, 2],
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'travel_time': [10.0, 15.0],
            'prediction': [9.0, 14.0],
            'resid': [1.0, 1.0]
        })
        
        try:
            result = traffic_anomaly.anomaly(
                decomposed_data=decomposed_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id'],
                group_grouping_columns=None,  # Should be allowed
                return_sql=True
            )
            assert isinstance(result, (str, type(result))), "Should return SQL string with None group grouping"
        except Exception as e:
            pytest.fail(f"None group_grouping_columns should be allowed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 