import pytest
import pandas as pd
import traffic_anomaly
from datetime import datetime

def test_decompose_with_extra_columns():
    """
    Test that decompose handles input DataFrames with extra columns gracefully.
    Regression test for issue #12.
    """
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
            rolling_window_enable=False # Disable rolling window to keep it simple, issue happens regardless
        )
    except ValueError as e:
        pytest.fail(f"decompose raised ValueError with extra columns: {e}")
    except Exception as e:
        pytest.fail(f"decompose raised unexpected exception: {e}")

    # Verify result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify extra columns are present (assuming default behavior preserves them or at least doesn't crash)
    # The current implementation of decompose (based on reading code) seems to preserve columns in 'table'
    # but filters rows.
    # However, if drop_extras=True (default), it drops specific calculation columns.
    # It does NOT explicitly drop unknown columns.
    assert 'group' in result.columns
    assert 'safe_col' in result.columns
