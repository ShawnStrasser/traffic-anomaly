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


def _calculate_changepoints_core(
    table: ibis.Expr,
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
) -> ibis.Expr:
    """Core changepoint calculation logic."""
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

        # Step 3: Optimized winsorized variance calculation using a single comprehensive join
        # Instead of 3 separate joins, we do one join and compute all variances together
        center = table_with_quantiles.alias('center')
        neighbor = table_with_quantiles.alias('neighbor')

        # Single join for all window relationships
        join_conditions_all = [
            neighbor[datetime_column] >= (center[datetime_column] - ibis.interval(days=rolling_window_days // 2)),
            neighbor[datetime_column] <= (center[datetime_column] + ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON)
        ]
        # Add grouping column conditions
        for col in grouping_columns:
            join_conditions_all.append(center[col] == neighbor[col])
            
        all_pairs = center.join(
            neighbor,
            join_conditions_all
        ).select(
            center_row=center['row_num'],
            center_date=center[datetime_column],
            neighbor_date=neighbor[datetime_column],
            **{f'center_{col}': center[col] for col in grouping_columns},
            # All quantiles needed for clipping
            left_lower=center['left_lower'],
            left_upper=center['left_upper'],
            right_lower=center['right_lower'],
            right_upper=center['right_upper'],
            combined_lower=center['combined_lower'],
            combined_upper=center['combined_upper'],
            neighbor_value=neighbor[value_column]
        ).mutate(
            # Determine which window each neighbor belongs to
            is_left=_['neighbor_date'] < _['center_date'],
            is_right=_['neighbor_date'] > _['center_date'],
            is_combined=_['neighbor_date'] != _['center_date']
        ).mutate(
            # Clip values for each window type using the new ibis.cases() function
            left_clipped=ibis.cases(
                (_['is_left'], _['neighbor_value'].clip(_['left_lower'], _['left_upper'])),
                else_=None
            ),
            right_clipped=ibis.cases(
                (_['is_right'], _['neighbor_value'].clip(_['right_lower'], _['right_upper'])),
                else_=None
            ),
            combined_clipped=ibis.cases(
                (_['is_combined'], _['neighbor_value'].clip(_['combined_lower'], _['combined_upper'])),
                else_=None
            )
        )
        
        # Aggregate variances for all windows in one operation
        all_variances = all_pairs.group_by(
            ['center_row', 'center_date'] + [f'center_{col}' for col in grouping_columns]
        ).aggregate(
            Left_Var=_['left_clipped'].var(),
            Right_Var=_['right_clipped'].var(),
            Combined_Var=_['combined_clipped'].var()
        )

        # Step 4: Join variance results back (single join instead of three)
        base_table = table_with_quantiles.drop([
            'left_lower', 'left_upper', 'right_lower', 'right_upper', 
            'combined_lower', 'combined_upper'
        ])

        # Single join with all variance results
        variance_join_conditions = [
            base_table['row_num'] == all_variances['center_row'],
            base_table[datetime_column] == all_variances['center_date']
        ]
        for col in grouping_columns:
            variance_join_conditions.append(base_table[col] == all_variances[f'center_{col}'])
            
        result = base_table.join(
            all_variances,
            variance_join_conditions,
            how='left'
        ).select(base_table, 'Left_Var', 'Right_Var', 'Combined_Var').drop('row_num')
        
    else:
        # Standard variance calculation
        result = table.mutate(
            Left_Var=_['lag_column'].var().over(left_window),
            Right_Var=_[value_column].var().over(right_window),
            Combined_Var=_[value_column].var().over(combined_window)
        )
    
    #####################################################################
    ##################### Calculate scores ##############################
    #####################################################################

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
    
    # Create windows for sample counting (reuse these later for averages)
    sample_window_before = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON,
        following=0
    )
    sample_window_after = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=0,
        following=ibis.interval(days=rolling_window_days // 2) - INTERVAL_EPSILON
    )
    
    result = result.mutate(
        sample_count_before=result[value_column].count().over(sample_window_before),
        sample_count_after=result[value_column].count().over(sample_window_after)
    ).mutate(
        score=(window_condition & 
               (_.sample_count_before >= min_samples) & 
               (_.sample_count_after >= min_samples)).ifelse(
            result['Combined_Cost'] - result['Left_Cost'] - result['Right_Cost'],
            0
        )
    # Add score lag column
    ).mutate(
        score_lag=_.score.lag(1).over(ibis.window(group_by=grouping_columns, order_by=datetime_column))
    ).drop('sample_count_before', 'sample_count_after')

    # Clean up intermediate columns and create scores table
    scores_table = result.drop([
        'Left_Cost', 'Right_Cost', 'Combined_Cost', 
        'Left_Var', 'Right_Var', 'Combined_Var', 'min_ts', 'max_ts'
    ]).order_by(_[datetime_column])

    # Create windows for peak detection
    peak_window = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=min_separation_days),
        following=ibis.interval(days=min_separation_days)
    )
    window_before_peak = ibis.window(
        group_by=grouping_columns,
        order_by=datetime_column,
        preceding=ibis.interval(days=min_separation_days)-INTERVAL_EPSILON, #added because values will be shited forward to remove the current row
        following=0
    )

    changepoints = scores_table.mutate(
        avg_before=(_['lag_column'].mean().over(sample_window_before)),
        avg_after=(_[value_column].mean().over(sample_window_after)),
        is_local_peak=(
            (_.score == _.score.max().over(peak_window)) & (_.score > score_threshold) &
            # Tie breaker that selects first row if there are multiple with the same max score
            (_.score > _.score_lag.max().over(window_before_peak))
        ),
    )

    # Filter to local peaks (score > 0 ensures both window boundary and min_samples were met)
    changepoints = changepoints.filter(changepoints['is_local_peak'] & (changepoints['score'] > 0))
    changepoints = changepoints.mutate(
        avg_diff=changepoints['avg_after'] - changepoints['avg_before'],
        pct_change=((changepoints['avg_after'] - changepoints['avg_before']) / 
                   ibis.greatest(changepoints['avg_before'].abs(), EPSILON) * 100)
    )

    # Select relevant columns
    final_result = changepoints.select(
        grouping_columns + [datetime_column, 'score', 'avg_before', 'avg_after', 'avg_diff', 'pct_change']
    ).order_by(grouping_columns + [datetime_column])
    
    return final_result


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
    min_samples: int = 30,
    return_sql: bool = False
) -> Union[ibis.Expr, Any, str]:  # ibis.Expr, pandas.DataFrame, or str
    """
    Detect changepoints in multivariate time series data using variance-based scoring.
    
    This function identifies changepoints by comparing variance in windows before and after
    each time point. It can use either standard variance or robust (Winsorized) variance.
    
    For optimal performance, when robust=True, the function first calculates changepoints using
    standard variance, then filters the dataset to only include regions around detected changepoints
    before performing the expensive robust variance calculation.
    
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
    min_samples : int, default 30
        Minimum number of samples required in both before and after windows for a changepoint score to be calculated.
        If this requirement is not met, the score is set to 0 rather than being calculated.
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

    # Performance optimization for robust variance calculation:
    # First calculate changepoints using standard variance, then filter data for robust calculation
    if robust:
        # Step 1: Get changepoints using standard variance (much faster)
        standard_changepoints = _calculate_changepoints_core(
            table, datetime_column, value_column, grouping_columns,
            rolling_window_days, False, upper_bound, lower_bound,
            score_threshold, min_separation_days, min_samples
        )
        
        # Step 2: For each entity with changepoints, get the timestamp range
        entity_ranges = standard_changepoints.group_by(grouping_columns).aggregate(
            min_changepoint_ts=standard_changepoints[datetime_column].min(),
            max_changepoint_ts=standard_changepoints[datetime_column].max()
        )
        
        # Step 3: Calculate buffer intervals
        buffer_interval = ibis.interval(days=rolling_window_days // 2 + min_separation_days)
        
        entity_ranges = entity_ranges.mutate(
            start_filter_ts=_.min_changepoint_ts - buffer_interval,
            end_filter_ts=_.max_changepoint_ts + buffer_interval
        )
        
        # Step 4: Filter original data to only regions around changepoints
        # If no changepoints exist, this join will result in an empty table naturally
        filtered_table = table.join(entity_ranges, grouping_columns, how='inner').filter(
            (_[datetime_column] >= _.start_filter_ts) & 
            (_[datetime_column] <= _.end_filter_ts)
        ).select(table)  # Keep only original table columns
        
        # Step 5: Run robust calculation on filtered data
        final_result = _calculate_changepoints_core(
            filtered_table, datetime_column, value_column, grouping_columns,
            rolling_window_days, True, upper_bound, lower_bound,
            score_threshold, min_separation_days, min_samples
        )
    else:
        # Standard variance calculation (no filtering needed)
        final_result = _calculate_changepoints_core(
            table, datetime_column, value_column, grouping_columns,
            rolling_window_days, False, upper_bound, lower_bound,
            score_threshold, min_separation_days, min_samples
        )
    
    # Return results based on parameters
    if return_sql:
        return ibis.to_sql(final_result)
    elif isinstance(data, ibis.Expr):
        return final_result  # Return Ibis expression directly if input was Ibis
    else:
        return final_result.execute()  # Convert to pandas (or similar) only for non-Ibis inputs