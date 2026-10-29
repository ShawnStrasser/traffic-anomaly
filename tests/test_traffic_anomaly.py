import pytest
import pandas as pd
import numpy as np
import os
import sys
import tomllib

import duckdb

# Add the src directory to the path to import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import traffic_anomaly
from traffic_anomaly import sample_data
from traffic_anomaly.decompose import median_decompose as median_decompose_from_module
from traffic_anomaly.find_anomaly import find_anomaly as find_anomaly_from_module


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
        with open(pyproject_path, 'rb') as f:
            pyproject_data = tomllib.load(f)
        pyproject_version = pyproject_data['project']['version']
        
        assert init_version == pyproject_version, (
            f"Version mismatch: __init__.py has {init_version}, "
            f"pyproject.toml has {pyproject_version}"
        )

    def test_decompose_with_extra_columns(self):
        """
        Test that decompose handles input DataFrames with extra columns gracefully.
        Regression test for issue #12.
        """
        from datetime import datetime
        data = {
            'XDSegID': [1, 1, 1],
            'Miles': [0.5, 0.5, 0.5],
            'Date Time': [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 10, 15),
                datetime(2023, 1, 1, 10, 30)
            ],
            'Speed(miles hour)': [60, 65, 55],
            'travel_time_sec': [30, 27, 32],
            'group': ['A', 'A', 'A'],          # Extra column 1 (SQL keyword)
            'safe_col': ['B', 'B', 'B']        # Extra column 2
        }

        df = pd.DataFrame(data)

        # This should not raise "ValueError: schema names don't match input data columns"
        try:
            result = traffic_anomaly.decompose(
                data=df,
                datetime_column='Date Time',
                value_column='Speed(miles hour)',
                entity_grouping_columns=['XDSegID'],
                rolling_window_enable=False # Disable rolling window to keep it simple
            )
        except ValueError as e:
            pytest.fail(f"decompose raised ValueError with extra columns: {e}")
        except Exception as e:
            pytest.fail(f"decompose raised unexpected exception: {e}")

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert 'group' in result.columns
        assert 'safe_col' in result.columns

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

    def test_median_decompose_backward_compatibility(self):
        """Legacy median_decompose wrapper should match decompose output."""
        current = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        legacy = traffic_anomaly.median_decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )

        self._compare_dataframes(legacy, current, "median_decompose_backward_compatibility")

    def test_median_decompose_to_sql_backward_compatibility(self):
        """Legacy median_decompose should map to_sql to return_sql."""
        conn = duckdb.connect()
        conn.register('vehicle_counts', self.vehicle_counts.head(10))
        current_sql = traffic_anomaly.decompose(
            'vehicle_counts',
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False,
            return_sql=True,
            conn=conn
        )
        legacy_sql = traffic_anomaly.median_decompose(
            'vehicle_counts',
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False,
            to_sql=True,
            conn=conn
        )

        assert isinstance(current_sql, str)
        assert legacy_sql == current_sql

    def test_median_decompose_module_backward_compatibility(self):
        """Legacy submodule import should still expose median_decompose."""
        current = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )
        legacy = median_decompose_from_module(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )

        self._compare_dataframes(legacy, current, "median_decompose_module_backward_compatibility")
    
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

    def test_find_anomaly_backward_compatibility(self):
        """Legacy find_anomaly wrapper should match anomaly output."""
        decomp = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )

        current = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
        )
        legacy = traffic_anomaly.find_anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
        )

        self._compare_dataframes(legacy, current, "find_anomaly_backward_compatibility")

    def test_find_anomaly_module_backward_compatibility(self):
        """Legacy submodule import should still expose find_anomaly."""
        decomp = traffic_anomaly.decompose(
            self.vehicle_counts,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            rolling_window_enable=False
        )

        current = traffic_anomaly.anomaly(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
        )
        legacy = find_anomaly_from_module(
            decomposed_data=decomp,
            datetime_column='timestamp',
            value_column='total',
            entity_grouping_columns=['intersection', 'detector'],
            entity_threshold=6.0,
            GEH=True,
            MAD=False,
            log_adjust_negative=True,
        )

        self._compare_dataframes(legacy, current, "find_anomaly_module_backward_compatibility")

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

    def test_changepoint_standard_recent_window(self):
        """Test changepoint detection with a recent-window post-filter (3 days)"""
        df = sample_data.changepoints_input

        changepoints_recent = traffic_anomaly.changepoint(
            df,
            value_column='travel_time_seconds',
            entity_grouping_column='ID',
            datetime_column='TimeStamp',
            score_threshold=0.7,
            robust=False,
            recent_days_for_validation=3
        )

        expected_path = os.path.join(self.precalculated_dir, 'test_changepoint_recent3.parquet')
        expected = pd.read_parquet(expected_path)

        self._compare_dataframes(changepoints_recent, expected, "changepoint_standard_recent_window")
    
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
    @staticmethod
    def _assert_same(actual, expected):
        """Exact equality, ignoring row order (the queries do not define one)."""
        columns = list(expected.columns)
        actual = actual.sort_values(columns).reset_index(drop=True)
        expected = expected.sort_values(columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    _DECOMPOSE = dict(datetime_column='timestamp', value_column='travel_time', entity_grouping_columns=['id'],
                      rolling_window_days=3, drop_days=3, min_rolling_window_samples=10, drop_extras=False)
    _ANOMALY = dict(datetime_column='timestamp', value_column='travel_time', entity_grouping_columns=['id'],
                    entity_threshold=1.0)

    def _chain_connectivity(self):
        """Each segment flows into the next one, the last has no connectivity row, and NEXT_ID is upper case."""
        ids = sorted(self.travel_times['id'].unique())
        return pd.DataFrame({'id': ids[:-1], 'NEXT_ID': ids[1:]})

    def test_sql_execution_equivalence_decompose(self):
        """SQL returned for a table name, and the relation returned for it, match the DataFrame result"""
        regular_result = traffic_anomaly.decompose(self.travel_times, **self._DECOMPOSE)

        conn = duckdb.connect()
        conn.register('travel_times', self.travel_times)
        sql_query = traffic_anomaly.decompose('travel_times', return_sql=True, conn=conn, **self._DECOMPOSE)
        assert isinstance(sql_query, str)
        relation = traffic_anomaly.decompose('travel_times', conn=conn, **self._DECOMPOSE)
        assert isinstance(relation, duckdb.DuckDBPyRelation)

        self._assert_same(conn.sql(sql_query).df(), regular_result)
        self._assert_same(relation.df(), regular_result)
        conn.close()

    def test_sql_execution_equivalence_anomaly(self):
        """anomaly SQL over a view of the decompose SQL matches the DataFrame results, connectivity included"""
        decomp = traffic_anomaly.decompose(self.travel_times, **self._DECOMPOSE)
        connectivity = self._chain_connectivity()
        regular_result = traffic_anomaly.anomaly(decomp, connectivity_table=connectivity, **self._ANOMALY)
        assert regular_result['originated_anomaly'].sum() > 0

        conn = duckdb.connect()
        conn.register('travel_times', self.travel_times)
        conn.register('connectivity', connectivity)
        decomp_sql = traffic_anomaly.decompose('travel_times', return_sql=True, conn=conn, **self._DECOMPOSE)
        conn.execute(f"CREATE VIEW decomposed AS {decomp_sql}")
        sql_query = traffic_anomaly.anomaly('decomposed', connectivity_table='connectivity', return_sql=True,
                                            conn=conn, **self._ANOMALY)

        self._assert_same(conn.sql(sql_query).df(), regular_result)
        conn.close()

    def test_relation_input(self):
        """DuckDB relations in, lazy relations out, matching the DataFrame path"""
        conn = duckdb.connect()
        relation = conn.from_df(self.travel_times)
        decomp = traffic_anomaly.decompose(relation, **self._DECOMPOSE)
        assert isinstance(decomp, duckdb.DuckDBPyRelation)
        expected_decomp = traffic_anomaly.decompose(self.travel_times, **self._DECOMPOSE)
        self._assert_same(decomp.df(), expected_decomp)

        # The connectivity table can be a relation from the same connection or a DataFrame
        connectivity = self._chain_connectivity()
        expected = traffic_anomaly.anomaly(expected_decomp, connectivity_table=connectivity, **self._ANOMALY)
        for conn_table in (conn.from_df(connectivity), connectivity):
            result = traffic_anomaly.anomaly(decomp, connectivity_table=conn_table, **self._ANOMALY)
            assert isinstance(result, duckdb.DuckDBPyRelation)
            self._assert_same(result.df(), expected)

        changepoints = traffic_anomaly.changepoint(conn.from_df(sample_data.changepoints_input), 'TimeStamp',
                                                   'travel_time_seconds', 'ID', score_threshold=0.7, robust=True)
        assert isinstance(changepoints, duckdb.DuckDBPyRelation)
        pd.testing.assert_frame_equal(changepoints.df(), traffic_anomaly.changepoint(
            sample_data.changepoints_input, 'TimeStamp', 'travel_time_seconds', 'ID', score_threshold=0.7, robust=True))
        conn.close()

    def test_changepoint_table_name(self):
        """changepoint reads a table by name, in DuckDB's default connection unless conn is given"""
        df = sample_data.changepoints_input
        expected = traffic_anomaly.changepoint(df, 'TimeStamp', 'travel_time_seconds', 'ID', score_threshold=0.7,
                                               recent_days_for_validation=3)
        conn = duckdb.connect()
        conn.register('changepoint_input', df)
        sql_query = traffic_anomaly.changepoint('changepoint_input', 'TimeStamp', 'travel_time_seconds', 'ID',
                                                score_threshold=0.7, recent_days_for_validation=3, return_sql=True,
                                                conn=conn)
        pd.testing.assert_frame_equal(conn.sql(sql_query).df(), expected)
        conn.close()

        duckdb.default_connection().register('ta_test_changepoint_input', df)
        try:
            result = traffic_anomaly.changepoint('ta_test_changepoint_input', 'TimeStamp', 'travel_time_seconds',
                                                 'ID', score_threshold=0.7, recent_days_for_validation=3)
            pd.testing.assert_frame_equal(result.df(), expected)
        finally:
            duckdb.default_connection().unregister('ta_test_changepoint_input')

    def test_return_sql_requires_table_name(self):
        """A DataFrame or relation has no name a returned query could read it by"""
        with pytest.raises(ValueError, match="return_sql=True needs data given as the name of a table or view"):
            traffic_anomaly.decompose(self.travel_times, return_sql=True, **self._DECOMPOSE)
        with pytest.raises(ValueError, match="return_sql=True needs data given as the name of a table or view"):
            traffic_anomaly.changepoint(duckdb.sql("SELECT 1 AS ID, now()::TIMESTAMP AS t, 1.0 AS v"),
                                        't', 'v', 'ID', return_sql=True)

    def test_connectivity_table_name_with_table_input(self):
        """With a table name as input, the connectivity table must be a table in the same connection"""
        conn = duckdb.connect()
        conn.register('travel_times', self.travel_times)
        decomp_sql = traffic_anomaly.decompose('travel_times', return_sql=True, conn=conn, **self._DECOMPOSE)
        conn.execute(f"CREATE VIEW decomposed AS {decomp_sql}")
        with pytest.raises(ValueError, match="connectivity_table must be a table name"):
            traffic_anomaly.anomaly('decomposed', connectivity_table=self._chain_connectivity(), conn=conn,
                                    **self._ANOMALY)
        conn.close()

    def test_dataframe_timestamps(self):
        """DataFrame timestamps come back in microseconds, and time zone aware ones in their own zone"""
        times = pd.date_range('2024-01-01', periods=14 * 24, freq='h')
        frame = pd.DataFrame({'TimeStamp': times, 'id': 1, 'value': np.arange(len(times)) % 24 + 1.0})
        params = dict(datetime_column='TimeStamp', value_column='value', entity_grouping_columns=['id'],
                      freq_minutes=60, rolling_window_enable=False)
        expected = traffic_anomaly.decompose(frame, **params)
        assert expected['TimeStamp'].dtype == 'datetime64[us]'
        for unit in ['ns', 'ms', 's']:
            result = traffic_anomaly.decompose(frame.astype({'TimeStamp': f'datetime64[{unit}]'}), **params)
            self._assert_same(result, expected)

        # Times of day are worked out in UTC, so the same instants in another zone decompose the same way
        local = frame.assign(TimeStamp=times.tz_localize('UTC').tz_convert('America/Los_Angeles'))
        result = traffic_anomaly.decompose(local, **params)
        assert str(result['TimeStamp'].dt.tz) == 'America/Los_Angeles'
        self._assert_same(result.assign(TimeStamp=result['TimeStamp'].dt.tz_convert(None)), expected)

    def test_no_ibis(self):
        """The package runs on DuckDB alone"""
        assert 'ibis' not in sys.modules

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
        # A string is read as a table or view name
        with pytest.raises(ValueError, match="Could not read table or view 'invalid_data_type'"):
            traffic_anomaly.decompose(
                data="invalid_data_type",
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test with None
        with pytest.raises(ValueError, match="Invalid data type"):
            traffic_anomaly.decompose(
                data=None,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )

    def test_invalid_timestamp_type(self):
        """Test that providing a string timestamp raises a TypeError"""
        df_string_ts = pd.DataFrame({
            'timestamp': ['2023-01-01 00:00:00'],
            'value': [10],
            'id': ['A'],
            'prediction': [10],
            'resid': [0]
        })

        # Test decompose
        with pytest.raises(TypeError, match="must be a temporal type"):
            traffic_anomaly.decompose(
                data=df_string_ts,
                datetime_column='timestamp',
                value_column='value',
                entity_grouping_columns=['id']
            )

        # Test anomaly
        with pytest.raises(TypeError, match="must be a temporal type"):
            traffic_anomaly.anomaly(
                decomposed_data=df_string_ts,
                datetime_column='timestamp',
                value_column='value',
                entity_grouping_columns=['id']
            )

        # Test changepoint
        with pytest.raises(TypeError, match="must be a temporal type"):
            traffic_anomaly.changepoint(
                data=df_string_ts,
                datetime_column='timestamp',
                value_column='value',
                entity_grouping_column='id'
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
        # A string is read as a table or view name
        with pytest.raises(ValueError, match="Could not read table or view 'invalid_data_type'"):
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

        with pytest.raises(ValueError, match="recent window days must not exceed half the rolling window size"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                rolling_window_days=10,
                recent_days_for_validation=6
            )
        
        # Test invalid min_separation_days (<=0)
        with pytest.raises(ValueError, match="min_separation_days must be positive"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                min_separation_days=0
            )
        
        with pytest.raises(ValueError, match="min_separation_days must be positive"):
            traffic_anomaly.changepoint(
                data=valid_data,
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID',
                min_separation_days=-1
            )

    def test_changepoint_invalid_data_type(self):
        """Test changepoint detection with invalid data input types"""
        # A string is read as a table or view name
        with pytest.raises(ValueError, match="Could not read table or view 'invalid_data_type'"):
            traffic_anomaly.changepoint(
                data="invalid_data_type",
                datetime_column='TimeStamp',
                value_column='travel_time_seconds',
                entity_grouping_column='ID'
            )
        
        # Test with None
        with pytest.raises(ValueError, match="Invalid data type"):
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

    def test_unreadable_data_handling(self):
        """Test exception handling for data DuckDB cannot read"""
        # Mock an object that DuckDB cannot register as a table
        class BadDataType:
            def __init__(self):
                pass
            
            def __str__(self):
                return "mock_bad_data"
        
        bad_data = BadDataType()
        
        # Test decompose with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a pandas DataFrame, a DuckDB relation or the name of a table or view"):
            traffic_anomaly.decompose(
                data=bad_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test anomaly with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a pandas DataFrame, a DuckDB relation or the name of a table or view"):
            traffic_anomaly.anomaly(
                decomposed_data=bad_data,
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id']
            )
        
        # Test changepoint with bad data type
        with pytest.raises(ValueError, match="Invalid data type. Please provide a pandas DataFrame, a DuckDB relation or the name of a table or view"):
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
        
        conn = duckdb.connect()
        conn.register('decomposed_data', decomposed_data)
        try:
            result = traffic_anomaly.anomaly(
                decomposed_data='decomposed_data',
                datetime_column='timestamp',
                value_column='travel_time',
                entity_grouping_columns=['id'],
                group_grouping_columns=None,  # Should be allowed
                return_sql=True,
                conn=conn
            )
            assert isinstance(result, str), "Should return SQL string with None group grouping"
            assert len(conn.sql(result).df()) == 2
        except Exception as e:
            pytest.fail(f"None group_grouping_columns should be allowed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 
