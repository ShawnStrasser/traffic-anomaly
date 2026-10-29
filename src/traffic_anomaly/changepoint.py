from typing import Union, List, Any

import duckdb

from ._sql import Source, quote, columns, number, interval

SECONDS_PER_DAY = 86400
EPSILON = '1e-06'


def _changepoints_sql(
    prefix: str,
    table: str,
    datetime_column: str,
    value_column: str,
    grouping_columns: List[str],
    rolling_window_days: int,
    robust: bool,
    upper_bound: float,
    lower_bound: float,
    score_threshold: float,
    min_separation_days: int,
    min_samples: int
) -> List[str]:
    """Common table expressions finding the changepoints in table. The last one, {prefix}changepoints, holds them."""
    p = prefix
    dt, value = quote(datetime_column), quote(value_column)
    group = columns(grouping_columns)
    on_group = lambda a, b: ' AND '.join(f'{a}.{quote(col)} = {b}.{quote(col)}' for col in grouping_columns)

    # Calculate intervals once for reuse throughout the function
    half_days = rolling_window_days // 2
    half_window = interval(half_days * SECONDS_PER_DAY - 1, 'SECOND')
    windows = f"""
        before AS (PARTITION BY {group} ORDER BY {dt} RANGE BETWEEN {half_window} PRECEDING AND CURRENT ROW),
        after AS (PARTITION BY {group} ORDER BY {dt} RANGE BETWEEN CURRENT ROW AND {half_window} FOLLOWING),
        combined AS (PARTITION BY {group} ORDER BY {dt}
            RANGE BETWEEN {interval(half_days, 'DAY')} PRECEDING AND {half_window} FOLLOWING)"""

    # Shift value_column forward for the left window
    ctes = [f"""
    {p}lagged AS (
        SELECT {group}, {dt}, {value}, LAG({value}) OVER (PARTITION BY {group} ORDER BY {dt}) AS lag_column
        FROM {table}
    )"""]

    if robust:
        # Winsorized variance: each window's values clipped to that window's quantiles
        lower, upper = number(lower_bound), number(upper_bound)
        def clipped(side, condition):
            return (f"CASE WHEN {condition} AND n.{value} IS NOT NULL "
                    f"THEN GREATEST(c.{side}_lower, LEAST(c.{side}_upper, n.{value})) END AS {side}_clipped")
        ctes.append(f"""
    {p}quantiles AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY {group} ORDER BY {dt}) AS row_num,
            QUANTILE_CONT(lag_column, {lower}) OVER before AS left_lower,
            QUANTILE_CONT(lag_column, {upper}) OVER before AS left_upper,
            QUANTILE_CONT({value}, {lower}) OVER after AS right_lower,
            QUANTILE_CONT({value}, {upper}) OVER after AS right_upper,
            QUANTILE_CONT({value}, {lower}) OVER combined AS combined_lower,
            QUANTILE_CONT({value}, {upper}) OVER combined AS combined_upper
        FROM {p}lagged
        WINDOW {windows}
    ),
    {p}robust_variances AS (
        SELECT row_num, {group}, {dt},
            VARIANCE(left_clipped) AS left_var,
            VARIANCE(right_clipped) AS right_var,
            VARIANCE(combined_clipped) AS combined_var
        FROM (
            SELECT c.row_num, {columns(grouping_columns, 'c')}, c.{dt},
                {clipped('left', f'n.{dt} < c.{dt}')},
                {clipped('right', f'n.{dt} > c.{dt}')},
                {clipped('combined', f'n.{dt} <> c.{dt}')}
            FROM {p}quantiles c
            JOIN {p}quantiles n
              ON n.{dt} >= c.{dt} - {interval(half_days, 'DAY')}
             AND n.{dt} <= c.{dt} + {half_window}
             AND {on_group('c', 'n')}
        ) pairs
        GROUP BY row_num, {group}, {dt}
    ),
    {p}variances AS (
        SELECT q.* EXCLUDE (row_num, left_lower, left_upper, right_lower, right_upper, combined_lower, combined_upper),
            v.left_var, v.right_var, v.combined_var
        FROM {p}quantiles q
        LEFT JOIN {p}robust_variances v ON q.row_num = v.row_num AND q.{dt} = v.{dt} AND {on_group('q', 'v')}
    )""")
    else:
        ctes.append(f"""
    {p}variances AS (
        SELECT *,
            VARIANCE(lag_column) OVER before AS left_var,
            VARIANCE({value}) OVER after AS right_var,
            VARIANCE({value}) OVER combined AS combined_var
        FROM {p}lagged
        WINDOW {windows}
    )""")

    # Score each point, leaving it at 0 unless both windows are complete and hold at least min_samples values
    min_separation = interval(min_separation_days, 'DAY')
    ctes.append(f"""
    {p}bounded AS (
        SELECT v.*, b.min_ts, b.max_ts
        FROM {p}variances v
        LEFT JOIN (
            SELECT {group}, MIN({dt}) AS min_ts, MAX({dt}) AS max_ts FROM {p}lagged GROUP BY {group}
        ) b ON {on_group('v', 'b')}
    ),
    {p}scores AS (
        SELECT * EXCLUDE (min_ts, max_ts),
            CASE WHEN {dt} >= min_ts + {interval(half_days, 'DAY')}
                  AND {dt} <= max_ts - {interval(half_days, 'DAY')}
                  AND COUNT({value}) OVER before >= {number(min_samples)}
                  AND COUNT({value}) OVER after >= {number(min_samples)}
                THEN (LN(combined_var + {EPSILON}) * 20 - LN(left_var + {EPSILON}) * 10) - LN(right_var + {EPSILON}) * 10
                ELSE 0
            END AS score
        FROM {p}bounded
        WINDOW {windows}
    ),
    {p}peaks AS (
        SELECT *,
            AVG(lag_column) OVER before AS avg_before,
            AVG({value}) OVER after AS avg_after,
            (score = MAX(score) OVER (PARTITION BY {group} ORDER BY {dt}
                    RANGE BETWEEN {min_separation} PRECEDING AND {min_separation} FOLLOWING))
                AND (score > {number(score_threshold)})
                -- Tie breaker that selects first row if there are multiple with the same max score
                AND (score > MAX(score_lag) OVER (PARTITION BY {group} ORDER BY {dt}
                    RANGE BETWEEN {interval(min_separation_days * SECONDS_PER_DAY - 1, 'SECOND')} PRECEDING AND CURRENT ROW))
                AS is_local_peak
        FROM (SELECT *, LAG(score) OVER (PARTITION BY {group} ORDER BY {dt}) AS score_lag FROM {p}scores)
        WINDOW {windows}
    ),
    {p}changepoints AS (
        -- score > 0 ensures both window boundary and min_samples were met
        SELECT {group}, {dt}, score, avg_before, avg_after,
            avg_after - avg_before AS avg_diff,
            (avg_after - avg_before) / GREATEST(ABS(avg_before), {EPSILON}) AS pct_change
        FROM {p}peaks
        WHERE is_local_peak AND score > 0
    )""")
    return ctes


def changepoint(
    data: Union[str, duckdb.DuckDBPyRelation, Any],  # table name, DuckDB relation or pandas DataFrame
    datetime_column: str,
    value_column: str,
    entity_grouping_column: Union[str, List[str]],
    rolling_window_days: int = 14,
    robust: bool = False,
    upper_bound: float = 0.95,
    lower_bound: float = 0.05,
    score_threshold: float = 5.0,
    min_separation_days: int = 3,
    min_samples: int = 30,
    recent_days_for_validation: int = 0,
    return_sql: bool = False,
    conn: duckdb.DuckDBPyConnection = None
) -> Union[duckdb.DuckDBPyRelation, Any, str]:  # DuckDB relation, pandas DataFrame, or str
    """
    Detect changepoints in multivariate time series data using variance-based scoring.

    This function identifies changepoints by comparing variance in windows before and after
    each time point. It can use either standard variance or robust (Winsorized) variance.

    For optimal performance, when robust=True, the function first calculates changepoints using
    standard variance, then filters the dataset to only include regions around detected changepoints
    before performing the expensive robust variance calculation.

    Parameters
    ----------
    data : str, DuckDB relation or pandas DataFrame
        Input data containing time series with entities: a pandas DataFrame (or anything else DuckDB can
        read), a DuckDB relation, or the name of a table or view in conn
    value_column : str
        Name of the column containing values to analyze
    entity_grouping_column : str or list of str
        Name(s) of the column(s) containing entity identifiers for grouping.
        Can be a single column name (str) or multiple column names (list of str).
    datetime_column : str
        Name of the column containing timestamps
    rolling_window_days : int, default 14
        Size of the rolling window in days (total window size, split between before/after)
    robust : bool, default False
        If True, use winsorized variance; if False, use standard variance
    upper_bound : float, default 0.95
        Upper quantile for winsorizing (only used when robust=True)
    lower_bound : float, default 0.05
        Lower quantile for winsorizing (only used when robust=True)
    score_threshold : float, default 5.0
        Minimum score threshold for identifying changepoints, increase this to decrease sensitivity
    min_separation_days : int, default 3
        Minimum separation between changepoints in days
    min_samples : int, default 30
        Minimum number of samples required in both before and after windows for a changepoint score to be calculated.
        If this requirement is not met, the score is set to 0 rather than being calculated.
    recent_days_for_validation : int, default 0
        Post-filter: length (in days) of the recent window per changepoint to average. Set to 0 to disable.
    return_sql : bool, default False
        If True, return the DuckDB SQL query instead of executing it. Requires data to be a table or view
        name; the query reads it by that name.
    conn : DuckDB connection, optional
        The connection holding the table or view named by data. Defaults to DuckDB's default connection.
        Not used for DataFrames or relations.

    Returns
    -------
    DuckDB relation, pandas DataFrame, or str
        If return_sql=True: SQL query string
        If data was a DataFrame: pandas DataFrame containing changepoints
        Otherwise: lazy DuckDB relation containing changepoints, in the same connection

        Changepoints table contains columns:
        - entity_grouping_column: Entity identifier(s) (same column name(s) as input)
        - datetime_column: Timestamp of changepoint
        - score: Changepoint score
        - avg_before: Average value before changepoint
        - avg_after: Average value after changepoint
        - avg_diff: Difference between after and before averages
        - pct_change: Percent change from before to after averages

    Raises
    ------
    ValueError
        If invalid data type or parameter values provided
    """
    # Parameter validation
    if not (0 <= upper_bound <= 1):
        raise ValueError("upper_bound must be between 0 and 1")
    if not (0 <= lower_bound <= 1):
        raise ValueError("lower_bound must be between 0 and 1")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be less than upper_bound")
    if rolling_window_days <= 0:
        raise ValueError("rolling_window_days must be positive")
    if min_separation_days <= 0:
        raise ValueError("min_separation_days must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    recent_window_days = recent_days_for_validation
    if recent_window_days < 0:
        raise ValueError("recent window days must be non-negative")
    if recent_window_days > rolling_window_days / 2:
        raise ValueError("recent window days must not exceed half the rolling window size")

    # Normalize entity_grouping_column to always be a list
    grouping_columns = [entity_grouping_column] if isinstance(entity_grouping_column, str) else entity_grouping_column

    source = Source(data, conn, return_sql, datetime_column, value_column, grouping_columns)

    dt, value = quote(datetime_column), quote(value_column)
    group = columns(grouping_columns)
    on_group = lambda a, b: ' AND '.join(f'{a}.{quote(col)} = {b}.{quote(col)}' for col in grouping_columns)
    params = (datetime_column, value_column, grouping_columns, rolling_window_days)
    thresholds = (upper_bound, lower_bound, score_threshold, min_separation_days, min_samples)

    # Performance optimization for robust variance calculation:
    # First calculate changepoints using standard variance, then filter data for robust calculation
    if robust:
        # Get changepoints using standard variance (much faster), then keep each entity's data from
        # a buffer before its first changepoint to a buffer after its last. Entities without
        # changepoints drop out.
        buffer = interval(rolling_window_days // 2 + min_separation_days, 'DAY')
        ctes = _changepoints_sql('standard_', source.table, *params, False, *thresholds)
        ctes.append(f"""
    entity_ranges AS (
        SELECT {group}, MIN({dt}) - {buffer} AS start_filter_ts, MAX({dt}) + {buffer} AS end_filter_ts
        FROM standard_changepoints
        GROUP BY {group}
    ),
    filtered AS (
        SELECT t.*
        FROM {source.table} t
        JOIN entity_ranges r ON {on_group('t', 'r')}
        WHERE t.{dt} >= r.start_filter_ts AND t.{dt} <= r.end_filter_ts
    )""")
        ctes += _changepoints_sql('', 'filtered', *params, True, *thresholds)
    else:
        ctes = _changepoints_sql('', source.table, *params, False, *thresholds)

    # Optional post-filter: drop changepoints whose recent mean is closer to the pre-change baseline
    if recent_window_days > 0:
        ctes.append(f"""
    recent_means AS (
        SELECT {columns(grouping_columns, 'cp')}, cp.{dt}, AVG(d.{value}) AS avg_recent
        FROM changepoints cp
        JOIN {source.table} d
          ON {on_group('cp', 'd')}
         AND d.{dt} >= cp.{dt}
         AND d.{dt} <= cp.{dt} + {interval(recent_window_days, 'DAY')}
        GROUP BY {columns(grouping_columns, 'cp')}, cp.{dt}
    )""")
        final = f"""
    SELECT cp.*
    FROM changepoints cp
    LEFT JOIN recent_means r ON {on_group('cp', 'r')} AND cp.{dt} = r.{dt}
    WHERE NOT COALESCE(ABS(r.avg_recent - cp.avg_before) < ABS(r.avg_recent - cp.avg_after), FALSE)
    ORDER BY {columns(grouping_columns, 'cp')}, cp.{dt}"""
    else:
        final = f"SELECT * FROM changepoints ORDER BY {group}, {dt}"

    sql = f"WITH {','.join(ctes)}\n{final}"
    return source.result(sql)
