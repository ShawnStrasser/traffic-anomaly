import ibis
from ibis import _
from typing import Union, List, Any
# Note: This function accepts ibis.Expr or pandas.DataFrame as input
# pandas is not required - ibis.memtable() can handle pandas DataFrames if pandas is available

INTERVAL_EPSILON = ibis.interval(seconds=1)
EPSILON = 1e-6

def _validate_columns(table: ibis.Expr, datetime_column: str, value_column: str, entity_grouping_column: Union[str, List[str]]) -> None:
    """Validate that required columns exist in the table."""
    missing_columns = []
    
    if datetime_column not in table.columns:
        missing_columns.append(datetime_column)
    if value_column not in table.columns:
        missing_columns.append(value_column)
    
    # Handle single string or list of strings for entity_grouping_column
    grouping_columns = [entity_grouping_column] if isinstance(entity_grouping_column, str) else entity_grouping_column
    for col in grouping_columns:
        if col not in table.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def changepoint(
    data: Union[ibis.Expr, Any],  # ibis.Expr or pandas.DataFrame
    datetime_column: str,
    value_column: str,
    entity_grouping_column: Union[str, List[str]],
    rolling_window_days: int = 14,
    robust: bool = False,
    upper_bound: float = 0.95,
    lower_bound: float = 0.05,
    score_threshold: float = 5.0,
    min_separation_days: int = 3,
    return_sql: bool = False
) -> Union[ibis.Expr, Any, str]:  # ibis.Expr, pandas.DataFrame, or str
    """
    Detect changepoints in multivariate time series data using variance-based scoring.
    
    This function identifies changepoints by comparing variance in windows before and after
    each time point. It can use either standard variance or robust (Winsorized) variance.
    
    Parameters
    ----------
    data : ibis.Expr or pandas.DataFrame
        Input data containing time series with entities
    value_column : str, default 'travel_time_seconds'
        Name of the column containing values to analyze
    entity_grouping_column : str or list of str, default 'ID'
        Name(s) of the column(s) containing entity identifiers for grouping.
        Can be a single column name (str) or multiple column names (list of str).
    datetime_column : str, default 'TimeStamp'
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
    return_sql : bool, default False
        If True, return SQL query string instead of executing
        
    Returns
    -------
    ibis.Expr, pandas.DataFrame, or str
        If return_sql=True: SQL query string
        If input was ibis.Expr: ibis.Expr containing changepoints
        If input was pandas.DataFrame: pandas.DataFrame containing changepoints
        
        Changepoints table contains columns:
        - entity_grouping_column: Entity identifier(s) (same column name(s) as input)
        - datetime_column: Timestamp of changepoint
        - score: Changepoint score
        - avg_before: Average value before changepoint
        - avg_after: Average value after changepoint
        - avg_diff: Difference between after and before averages
        
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
    if min_separation_days < 0:
        raise ValueError("min_separation_days must be non-negative")
    
    # Check if data is an Ibis table
    if isinstance(data, ibis.Expr):
        table = data
    else:
        try:
            table = ibis.memtable(data)
        except Exception as e:
            raise ValueError('Invalid data type. Please provide a valid Ibis table or pandas DataFrame.')
    
    # Validate required columns exist
    _validate_columns(table, datetime_column, value_column, entity_grouping_column)
    
    # Normalize entity_grouping_column to always be a list
    grouping_columns = [entity_grouping_column] if isinstance(entity_grouping_column, str) else entity_grouping_column

    # Shift value_column forward for the left window
    table = table.mutate(
        lag_column=_[value_column].lag(1).over(
            ibis.window(group_by=grouping_columns, order_by=datetime_column)
        )
    )

    # Create windows for each variance calculation
    left_window = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON,  # added because values will be shifted forward to remove the current row
        following=0
    )

    right_window = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=0,
        following=ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON
    )

    combined_window = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=rolling_window_days // 2),
        following=ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON
    )
    
    if robust:
        # Winsorized variance calculation
        # Step 1: Add row numbers for joining
        table_with_rn = table.mutate(
            row_num=ibis.row_number().over(
                ibis.window(group_by=grouping_columns, order_by=datetime_column)
            )
        )

        # Step 2: Calculate quantiles for each window
        table_with_quantiles = table_with_rn.mutate(
            # Left window quantiles
            left_lower=_['lag_column'].quantile(lower_bound).over(left_window),
            left_upper=_['lag_column'].quantile(upper_bound).over(left_window),
            
            # Right window quantiles  
            right_lower=_[value_column].quantile(lower_bound).over(right_window),
            right_upper=_[value_column].quantile(upper_bound).over(right_window),
            
            # Combined window quantiles
            combined_lower=_[value_column].quantile(lower_bound).over(combined_window),
            combined_upper=_[value_column].quantile(upper_bound).over(combined_window)
        )

        # Step 3: Calculate winsorized variances using joins
        center = table_with_quantiles.alias('center')
        neighbor = table_with_quantiles.alias('neighbor')

        # Left variance
        join_conditions_left = [
            neighbor[datetime_column] >= (center[datetime_column] - ibis.interval(days=rolling_window_days // 2)),
            neighbor[datetime_column] < center[datetime_column]
        ]
        # Add grouping column conditions
        for col in grouping_columns:
            join_conditions_left.append(center[col] == neighbor[col])
            
        left_pairs = center.join(
            neighbor,
            join_conditions_left
        ).select(
            center_row=center['row_num'],
            center_date=center[datetime_column],
            **{f'center_{col}': center[col] for col in grouping_columns},
            center_lower=center['left_lower'],
            center_upper=center['left_upper'],
            neighbor_value=neighbor[value_column]
        )
        left_variance = left_pairs.mutate(
            clipped_neighbor_value=left_pairs['neighbor_value'].clip(left_pairs['center_lower'], left_pairs['center_upper'])
        ).group_by(['center_row', 'center_date'] + [f'center_{col}' for col in grouping_columns]).aggregate(
            Left_Var=_['clipped_neighbor_value'].var()
        )

        # Right variance
        join_conditions_right = [
            neighbor[datetime_column] > center[datetime_column],
            neighbor[datetime_column] <= (center[datetime_column] + ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON)
        ]
        # Add grouping column conditions
        for col in grouping_columns:
            join_conditions_right.append(center[col] == neighbor[col])
            
        right_pairs = center.join(
            neighbor,
            join_conditions_right
        ).select(
            center_row=center['row_num'],
            center_date=center[datetime_column],
            **{f'center_{col}': center[col] for col in grouping_columns},
            center_lower=center['right_lower'],
            center_upper=center['right_upper'],
            neighbor_value=neighbor[value_column]
        )
        right_variance = right_pairs.mutate(
            clipped_neighbor_value=right_pairs['neighbor_value'].clip(right_pairs['center_lower'], right_pairs['center_upper'])
        ).group_by(['center_row', 'center_date'] + [f'center_{col}' for col in grouping_columns]).aggregate(
            Right_Var=_['clipped_neighbor_value'].var()
        )

        # Combined variance
        join_conditions_combined = [
            neighbor[datetime_column] >= (center[datetime_column] - ibis.interval(days=rolling_window_days // 2)),
            neighbor[datetime_column] <= (center[datetime_column] + ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON),
            neighbor[datetime_column] != center[datetime_column]
        ]
        # Add grouping column conditions
        for col in grouping_columns:
            join_conditions_combined.append(center[col] == neighbor[col])
            
        combined_pairs = center.join(
            neighbor,
            join_conditions_combined
        ).select(
            center_row=center['row_num'],
            center_date=center[datetime_column],
            **{f'center_{col}': center[col] for col in grouping_columns},
            center_lower=center['combined_lower'],
            center_upper=center['combined_upper'],
            neighbor_value=neighbor[value_column]
        )
        combined_variance = combined_pairs.mutate(
            clipped_neighbor_value=combined_pairs['neighbor_value'].clip(combined_pairs['center_lower'], combined_pairs['center_upper'])
        ).group_by(['center_row', 'center_date'] + [f'center_{col}' for col in grouping_columns]).aggregate(
            Combined_Var=_['clipped_neighbor_value'].var()
        )

        # Step 4: Join all variance results back
        base_table = table_with_quantiles.drop(['left_lower', 'left_upper', 'right_lower', 'right_upper', 'combined_lower', 'combined_upper'])

        # Join with left_variance
        left_join_conditions = [
            base_table['row_num'] == left_variance['center_row'],
            base_table[datetime_column] == left_variance['center_date']
        ]
        for col in grouping_columns:
            left_join_conditions.append(base_table[col] == left_variance[f'center_{col}'])
            
        result = base_table.join(
            left_variance,
            left_join_conditions,
            how='left'
        ).select(base_table, 'Left_Var')

        # Join with right_variance
        right_join_conditions = [
            result['row_num'] == right_variance['center_row'],
            result[datetime_column] == right_variance['center_date']
        ]
        for col in grouping_columns:
            right_join_conditions.append(result[col] == right_variance[f'center_{col}'])
            
        result = result.join(
            right_variance,
            right_join_conditions,
            how='left'
        ).select(result, 'Right_Var')

        # Join with combined_variance
        combined_join_conditions = [
            result['row_num'] == combined_variance['center_row'],
            result[datetime_column] == combined_variance['center_date']
        ]
        for col in grouping_columns:
            combined_join_conditions.append(result[col] == combined_variance[f'center_{col}'])
            
        result = result.join(
            combined_variance,
            combined_join_conditions,
            how='left'
        ).select(result, 'Combined_Var')

        # Drop row_num column
        result = result.drop('row_num')
        
    else:
        # Standard variance calculation
        result = table.mutate(
            Left_Var=_['lag_column'].var().over(left_window),
            Right_Var=_[value_column].var().over(right_window),
            Combined_Var=_[value_column].var().over(combined_window)
        )
    
    # Calculate scores
    # Calculate min and max timestamps for each entity to identify window boundaries
    entity_bounds = table.group_by(grouping_columns).aggregate(
        min_ts=table[datetime_column].min(),
        max_ts=table[datetime_column].max()
    )

    # Join entity bounds back to the results
    joined_result = result.join(entity_bounds, grouping_columns, how='left')
    result = joined_result.select(result, 'min_ts', 'max_ts')

    # Add cost and score columns, making score NaN if the window is incomplete
    half_window_interval = ibis.interval(days=rolling_window_days // 2)
    result = result.mutate(
        Combined_Cost=(20 * (result['Combined_Var'] + EPSILON).ln()),
        Left_Cost=(10 * (result['Left_Var'] + EPSILON).ln()),
        Right_Cost=(10 * (result['Right_Var'] + EPSILON).ln())
    )
    
    # Check window boundaries to ensure we have complete windows
    window_condition = (
        (result[datetime_column] >= result['min_ts'] + half_window_interval) &
        (result[datetime_column] <= result['max_ts'] - half_window_interval)
    )
    
    result = result.mutate(
        score=window_condition.ifelse(
            result['Combined_Cost'] - result['Left_Cost'] - result['Right_Cost'],
            None
        )
    )

    # Clean up intermediate columns and create scores table
    scores_table = result.drop([
        'Left_Cost', 'Right_Cost', 'Combined_Cost', 
        'Left_Var', 'Right_Var', 'Combined_Var', 'min_ts', 'max_ts'
    ]).order_by(_[datetime_column])
    
    # Identify changepoints
    table_for_changepoints = scores_table.filter(_["score"].notnull())

    # Create windows for peak detection
    peak_window = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=min_separation_days),
        following=ibis.interval(days=min_separation_days)
    )
    window_before = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=min_separation_days)-INTERVAL_EPSILON, #added because values will be shited forward to remove the current row
        following=0
    )
    window_after = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=0,
        following=ibis.interval(days=min_separation_days)
    )

    changepoints = table_for_changepoints.mutate(
        is_local_peak=((_.score == _.score.max().over(peak_window)) & (_.score > score_threshold)),
    avg_before=(_['lag_column'].mean().over(window_before)),
        avg_after=(_[value_column].mean().over(window_after)),
    )

    # Filter to local peaks
    changepoints = changepoints.filter(changepoints['is_local_peak'])
    changepoints = changepoints.mutate(
        avg_diff=changepoints['avg_after'] - changepoints['avg_before']
    )

    # Select relevant columns
    final_result = changepoints.select(
        grouping_columns + [datetime_column, 'score', 'avg_before', 'avg_after', 'avg_diff']
    ).order_by(grouping_columns + [datetime_column])
    
    # Return results based on parameters
    if return_sql:
        return ibis.to_sql(final_result)
    elif isinstance(data, ibis.Expr):
        return final_result  # Return Ibis expression directly if input was Ibis
    else:
        return final_result.execute()  # Convert to pandas (or similar) only for non-Ibis inputs