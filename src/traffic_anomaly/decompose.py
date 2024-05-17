import ibis
from ibis import _

def median_decompose(data,
                     datetime_column='timestamp',
                     value_column='value',
                     entity_grouping_columns=['id'],
                     freq_minutes=15,
                     rolling_window_days=7,
                     drop_days=7,
                     min_rolling_window_samples=96*5,
                     min_time_of_day_samples=7,
                     rolling_window_enable=True, # False disables all rolling functions including drop days and uses static median
                     drop_extras=True,
                     to_sql=False
                     ):
    """
    Decomposes a time series dataset into rolling median, seasonal (day and week), and residual components.

    Args:
        data (pd.DataFrame or ibis.expr.types.TableExpr): The time series data to decompose.
        datetime_column (str): The name of the column containing datetime information. Default is 'timestamp'.
        value_column (str): The name of the column containing the values to be decomposed. Default is 'value'.
        entity_grouping_columns (list): List of column names to group by. Default is ['id'].
        freq_minutes (int): Frequency of the time series data in minutes. Default is 15.
        rolling_window_days (int): Number of days to use for the rolling window. Default is 7.
        drop_days (int): Number of days to drop from the beginning of the dataset. Default is 7.
        min_rolling_window_samples (int): Minimum number of samples required in the rolling window. Default is 96*5.
        min_time_of_day_samples (int): Minimum number of samples required for each time of day. Default is 7.
        rolling_window_enable (bool): Whether to enable the rolling window functionality. Default is True.
                                      If False, disables all rolling functions, including drop days, and uses a static median.
        drop_extras (bool): Whether to drop extra columns from the result. Default is True.
        to_sql (bool): Whether to return the result as an SQL query string. Default is False.

    Returns:
        pd.DataFrame: The decomposed time series data with the following columns:
            - entity_grouping_columns: The columns used for grouping.
            - datetime_column: The datetime column.
            - value_column: The original value column.
            - median: The rolling median component.
            - season_day: The daily seasonal component.
            - season_week: The weekly seasonal component.
            - resid: The residual component.
            - prediction: The predicted value (median + season_day + season_week).
        If to_sql is True, returns the SQL query string instead.

    Notes:
        - The function supports both pandas DataFrame and Ibis table expressions as input.
        - The rolling window functionality can be disabled by setting rolling_window_enable to False.
        - If rolling_window_enable is True, the function filters out rows with insufficient samples in the rolling window
          and drops the specified number of days from the beginning of the dataset.
        - The function calculates the daily and weekly seasonal components using the median of the detrended values.
        - The residual component is calculated by subtracting the median, daily, and weekly seasonal components from the detrended values.
        - The predicted value is calculated as the sum of the median, daily, and weekly seasonal components, with a minimum value of 0.
        - If drop_extras is True, the function drops the extra columns (median, season_day, season_week) from the result.
    """
    # Check if df_or_table is an Ibis table
    if isinstance(data, ibis.Expr):
        table = data
    else:
        try:
            table = ibis.memtable(data)
        except Exception as e:
            raise ValueError('Invalid data type. Please provide a valid Ibis table or Pandas DataFrame.')

    window = ibis.window(
        group_by=entity_grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(hours=(24 * rolling_window_days) - 1),
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

    if to_sql:
        return ibis.to_sql(result)
    else:
        return result.execute()
    