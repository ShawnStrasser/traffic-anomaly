import ibis
from ibis import _

def median_decompose(data,
                     freq_minutes,
                     rolling_window_days,
                     drop_days=0,
                     min_rolling_window_samples=0,
                     min_time_of_day_samples=0,
                     return_sql=False
                     ):
    """
    Decomposes a time series dataset into rolling median, seasonal (day and week), and residual components.

    Args:
        data (pd.DataFrame or ibis.expr.types.TableExpr): The time series data to decompose.
        freq_minutes (int): Frequency of the time series data in minutes. Default is 60.
        rolling_window_days (int): Number of days to use for the rolling window. Default is 7.
        drop_days (int): Number of days to drop from the beginning of the dataset. Default is 7.
        min_rolling_window_samples (int): Minimum number of samples required in the rolling window. Default is 56.
        min_time_of_day_samples (int): Minimum number of samples required for each time of day. Default is 7.

    Returns:
        pd.DataFrame: The decomposed time series data. If return_sql is True, returns the SQL query string instead.

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
        group_by=table.id,
        order_by=table.timestamp,
        preceding=ibis.interval(hours=(24 * rolling_window_days) - 1),
        following=0
    )

    result = (
        table
        .mutate(
            rolling_row_count=_.count().over(window).cast('int16'),
            median=_.value.median().over(window).cast('float32')
        )
        .filter(_.timestamp >= _.timestamp.min() + ibis.interval(days=drop_days))
        .mutate(
            detrend=_.value - _.median,
            time_of_day=((_.timestamp.hour() * 60 + _.timestamp.minute()) / freq_minutes + 1).cast('int16'),
            day_of_week=_.timestamp.day_of_week.index(),
        )
        .group_by([_.id, _.time_of_day])
        .mutate(season_day=_.detrend.median().cast('float32'), time_of_day_count=_.count().cast('int16'))
        .group_by([_.id, _.day_of_week, _.time_of_day])
        .mutate(season_week=(_.detrend - _.season_day).median().cast('float32'))
        .mutate(resid=_.detrend - _.season_day - _.season_week)
        .filter(_.rolling_row_count >= min_rolling_window_samples, _.time_of_day_count >= min_time_of_day_samples)
        .select(
            _.timestamp,
            _.id,
            _.value,
            _.median,
            _.season_day,
            _.season_week,
            _.resid,
        )
        .order_by([_.timestamp, _.id])
    )

    if return_sql:
        return ibis.to_sql(result)
    else:
        return result.execute()
    