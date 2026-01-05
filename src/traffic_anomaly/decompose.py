import ibis
from ibis import _
from typing import Union, List, Any
# Note: This function accepts ibis.Expr or pandas.DataFrame as input
# pandas is not required - ibis.memtable() can handle pandas DataFrames if pandas is available


def _validate_columns(table: ibis.Expr, datetime_column: str, value_column: str, entity_grouping_columns: List[str]) -> None:
    """Validate that required columns exist in the table."""
    missing_columns = []
    
    if datetime_column not in table.columns:
        missing_columns.append(datetime_column)
    if value_column not in table.columns:
        missing_columns.append(value_column)
    
    for col in entity_grouping_columns:
        if col not in table.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not table[datetime_column].type().is_temporal():
        raise TypeError(f"Column '{datetime_column}' must be a temporal type (e.g., timestamp or date), but is {table[datetime_column].type()}")


def decompose(
    data: Union[ibis.Expr, Any],  # ibis.Expr or pandas.DataFrame
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
    dialect = None
) -> Union[ibis.Expr, Any, str]:  # ibis.Expr, pandas.DataFrame, or str
    """
    Decomposes a time series dataset into rolling median, seasonal (day and week), and residual components.

    Args:
        data: The time series data to decompose. Must be an ibis.Expr or pandas.DataFrame.
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
        return_sql: Whether to return the result as an SQL query string.
        dialect: Option to output a specific SQL dialect when return_sql=True

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
        If return_sql is True, returns the SQL query string instead.
        Returns ibis.Expr if input was ibis.Expr, pandas.DataFrame if input was pandas.DataFrame.

    Notes:
        - The function supports both pandas DataFrame and Ibis table expressions as input.
        - pandas is not required as a dependency - ibis.memtable() can handle pandas DataFrames if pandas is available.
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
    
    # Check if data is an Ibis table
    if isinstance(data, ibis.Expr):
        table = data
    else:
        # Assume it's a pandas DataFrame or similar tabular data
        try:
            table = ibis.memtable(data)
        except Exception as e:
            raise ValueError('Invalid data type. Please provide a valid Ibis table or pandas DataFrame.')

    # Validate required columns exist
    _validate_columns(table, datetime_column, value_column, entity_grouping_columns)

    window = ibis.window(
        group_by=entity_grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=rolling_window_days),
        following=0
    )

    if rolling_window_enable:
        result = (
            table
            .mutate(
                rolling_row_count=_.count().over(window).cast('int16'),
                median=_[value_column].median().over(window).cast('float32')
            )
            .filter(_[datetime_column] >= _[datetime_column].min() + ibis.interval(days=drop_days))
        )
    else:
        result = (
            table
            .group_by(entity_grouping_columns)
            .mutate(median=_[value_column].median().cast('float32'))
        )

    result = (
        result
        .mutate(
            detrend=_[value_column] - _.median,
            time_of_day=((_[datetime_column].hour() * 60 + _[datetime_column].minute()) / freq_minutes + 1).cast('int16'),
            day_of_week=_[datetime_column].day_of_week.index(),
        )
        .group_by(entity_grouping_columns + [_.time_of_day])
        .mutate(season_day=_.detrend.median().cast('float32'),
                time_of_day_count=_.count().cast('int16'))
        .group_by(entity_grouping_columns + [_.day_of_week, _.time_of_day])
        .mutate(season_week=(_.detrend - _.season_day).median().cast('float32'))
        .mutate(resid=_.detrend - _.season_day - _.season_week,
                prediction=ibis.greatest(_.median + _.season_day + _.season_week, 0))
    )

    if rolling_window_enable:
        result = (
            result
            .filter(_.rolling_row_count >= min_rolling_window_samples)
            .drop('rolling_row_count')
        )
    
    result = (
        result
        .filter(_.time_of_day_count >= min_time_of_day_samples)
        .drop('detrend', 'time_of_day', 'day_of_week', 'time_of_day_count')
    )

    if drop_extras:
        result = result.drop('median', 'season_day', 'season_week')

    if return_sql:
        return ibis.to_sql(result, dialect=dialect)
    elif isinstance(data, ibis.Expr):
        return result  # Return Ibis expression directly if input was Ibis
    else:
        return result.execute()  # Convert to pandas (or similar) only for non-Ibis inputs
    