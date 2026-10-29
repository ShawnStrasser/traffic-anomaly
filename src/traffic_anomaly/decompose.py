from typing import Union, List, Any
import warnings

import duckdb

from ._sql import Source, quote, columns, number, interval

# Components added to the output, in order. median, season_day and season_week are dropped by drop_extras.
_COMPONENTS = {
    'median': '__ta_median',
    'season_day': '__ta_season_day',
    'season_week': '__ta_season_week',
    'resid': '(__ta_detrend - __ta_season_day) - __ta_season_week',
    'prediction': 'GREATEST(__ta_median + __ta_season_day + __ta_season_week, 0)',
}
_EXTRAS = ('median', 'season_day', 'season_week')


def decompose(
    data: Union[str, duckdb.DuckDBPyRelation, Any],  # table name, DuckDB relation or pandas DataFrame
    datetime_column: str = 'timestamp',
    value_column: str = 'value',
    entity_grouping_columns: List[str] = None,
    freq_minutes: int = 15,
    rolling_window_days: int = 7,
    drop_days: int = 7,
    min_rolling_window_samples: int = 96*5,
    min_time_of_day_samples: int = 7,
    rolling_window_enable: bool = True,
    drop_extras: bool = True,
    return_sql: bool = False,
    conn: duckdb.DuckDBPyConnection = None
) -> Union[duckdb.DuckDBPyRelation, Any, str]:  # DuckDB relation, pandas DataFrame, or str
    """
    Decomposes a time series dataset into rolling median, seasonal (day and week), and residual components.

    Args:
        data: The time series data to decompose. A pandas DataFrame (or anything else DuckDB can read, such as
              an Arrow table), a DuckDB relation, or the name of a table or view in conn.
        datetime_column: The name of the column containing datetime information.
        value_column: The name of the column containing the values to be decomposed.
        entity_grouping_columns: List of column names to group by.
        freq_minutes: Frequency of the time series data in minutes.
        rolling_window_days: Number of days to use for the rolling window.
        drop_days: Number of days to drop from the beginning of the dataset.
        min_rolling_window_samples: Minimum number of samples required in the rolling window.
        min_time_of_day_samples: Minimum number of samples required for each time of day.
        rolling_window_enable: Whether to enable the rolling window functionality.
                              If False, disables all rolling functions, including drop days, and uses a static median.
        drop_extras: Whether to drop extra columns from the result.
        return_sql: Whether to return the DuckDB SQL query instead of the result. Requires data to be a table
                    or view name; the query reads it by that name.
        conn: The DuckDB connection holding the table or view named by data. Defaults to DuckDB's default
              connection (the one duckdb.sql() uses). Not used for DataFrames or relations.

    Returns:
        The decomposed time series data with the following columns:
            - entity_grouping_columns: The columns used for grouping.
            - datetime_column: The datetime column.
            - value_column: The original value column.
            - median: The rolling median component.
            - season_day: The daily seasonal component.
            - season_week: The weekly seasonal component.
            - resid: The residual component.
            - prediction: The predicted value (median + season_day + season_week).
        A pandas DataFrame if data was a DataFrame, otherwise a lazy DuckDB relation in the same connection.
        If return_sql is True, returns the SQL query string instead.

    Notes:
        - The rolling window functionality can be disabled by setting rolling_window_enable to False.
        - If rolling_window_enable is True, the function filters out rows with insufficient samples in the rolling window
          and drops the specified number of days from the beginning of the dataset.
        - The function calculates the daily and weekly seasonal components using the median of the detrended values.
        - The residual component is calculated by subtracting the median, daily, and weekly seasonal components from the detrended values.
        - The predicted value is calculated as the sum of the median, daily, and weekly seasonal components, with a minimum value of 0.
        - If drop_extras is True, the function drops the extra columns (median, season_day, season_week) from the result.
    """
    # Set default for mutable parameter
    if entity_grouping_columns is None:
        entity_grouping_columns = ['id']

    # Validate parameter types
    if not isinstance(entity_grouping_columns, list):
        raise ValueError('entity_grouping_columns must be a list.')

    source = Source(data, conn, return_sql, datetime_column, value_column, entity_grouping_columns)

    dt, value = quote(datetime_column), quote(value_column)
    entity = columns(entity_grouping_columns)

    if rolling_window_enable:
        trend = f"""
        trend AS (
            SELECT *,
                COUNT(*) OVER rolling AS __ta_rolling_row_count,
                CAST(QUANTILE_CONT({value}, 0.5) OVER rolling AS REAL) AS __ta_median
            FROM {source.table}
            WINDOW rolling AS (
                PARTITION BY {entity} ORDER BY {dt}
                RANGE BETWEEN {interval(rolling_window_days, 'DAY')} PRECEDING AND CURRENT ROW
            )
        ),
        trimmed AS (
            SELECT * FROM trend
            WHERE {dt} >= (SELECT MIN({dt}) FROM trend) + {interval(drop_days, 'DAY')}
        ),"""
        trended = 'trimmed'
        keep = f'__ta_rolling_row_count >= {number(min_rolling_window_samples)} AND '
    else:
        trend = f"""
        trend AS (
            SELECT *, CAST(QUANTILE_CONT({value}, 0.5) OVER (PARTITION BY {entity}) AS REAL) AS __ta_median
            FROM {source.table}
        ),"""
        trended = 'trend'
        keep = ''

    # Input columns keep their place (a component with the same name replaces them), new components follow
    select = []
    for col in source.columns:
        if drop_extras and col in _EXTRAS:
            continue
        select.append(f'{_COMPONENTS[col]} AS {quote(col)}' if col in _COMPONENTS else quote(col))
    for col, expr in _COMPONENTS.items():
        if col not in source.columns and not (drop_extras and col in _EXTRAS):
            select.append(f'{expr} AS {quote(col)}')

    sql = f"""
    WITH {trend}
    detrended AS (
        SELECT *,
            {value} - __ta_median AS __ta_detrend,
            CAST((EXTRACT(hour FROM {dt}) * 60 + EXTRACT(minute FROM {dt})) / {number(freq_minutes)} + 1 AS SMALLINT)
                AS __ta_time_of_day,
            (DAYOFWEEK({dt}) + 6) % 7 AS __ta_day_of_week
        FROM {trended}
    ),
    season_day AS (
        SELECT *,
            CAST(QUANTILE_CONT(__ta_detrend, 0.5) OVER time_of_day AS REAL) AS __ta_season_day,
            COUNT(*) OVER time_of_day AS __ta_time_of_day_count
        FROM detrended
        WINDOW time_of_day AS (PARTITION BY {entity}, __ta_time_of_day)
    ),
    season_week AS (
        SELECT *,
            CAST(QUANTILE_CONT(__ta_detrend - __ta_season_day, 0.5)
                OVER (PARTITION BY {entity}, __ta_day_of_week, __ta_time_of_day) AS REAL) AS __ta_season_week
        FROM season_day
    )
    SELECT {', '.join(select)}
    FROM season_week
    WHERE {keep}__ta_time_of_day_count >= {number(min_time_of_day_samples)}
    """
    return source.result(sql)


def median_decompose(*args, to_sql: bool = False, **kwargs):
    """Backward-compatible wrapper for the pre-v2 decompose entry point."""
    warnings.warn(
        "traffic_anomaly.decompose.median_decompose() is deprecated; use traffic_anomaly.decompose.decompose() or traffic_anomaly.decompose() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if 'return_sql' in kwargs:
        raise TypeError("median_decompose() received both 'to_sql' and 'return_sql'. Use only one.")
    return decompose(*args, return_sql=to_sql, **kwargs)
