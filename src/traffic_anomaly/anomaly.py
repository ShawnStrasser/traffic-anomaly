import ibis
from ibis import _
from typing import Union, List, Optional, Any
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


def anomaly(
    decomposed_data: Union[ibis.Expr, Any],  # ibis.Expr or pandas.DataFrame
    datetime_column: str,
    value_column: str,
    entity_grouping_columns: List[str],
    group_grouping_columns: Optional[List[str]] = None,
    entity_threshold: float = 3.5,
    group_threshold: float = 3.5,
    GEH: bool = False,
    MAD: bool = False,
    log_adjust_negative: bool = False,
    connectivity_table: Optional[Union[ibis.Expr, Any]] = None,
    return_sql: bool = False,
    dialect = None
) -> Union[ibis.Expr, Any, str]:  # ibis.Expr, pandas.DataFrame, or str
    """
    Detect anomalies in the time series data at both entity and group levels.

    Args:
        decomposed_data: The decomposed time series data. Must be an ibis.Expr or pandas.DataFrame.
        datetime_column: The name of the column containing datetime information.
        value_column: The name of the column containing the actual values.
        entity_grouping_columns: The list of columns used for grouping entities.
        group_grouping_columns: The list of columns used for grouping groups.
        entity_threshold: The threshold for detecting anomalies at the entity level.
        group_threshold: The threshold for detecting anomalies at the group level.
        GEH: Whether to use GEH scores for entity-level anomaly detection.
        MAD: Whether to use Median Absolute Deviation (MAD) for group-level anomaly detection.
        log_adjust_negative: Whether to make negative residuals more extreme for data censored at 0.
        connectivity_table: Optional table to calculate originated anomalies. 
                           When provided, only a single entity_grouping_column is supported.
        return_sql: Whether to return the SQL query string instead of the result.
        dialect: Option to output a specific SQL dialect when return_sql=True

    Returns:
        The detected anomalies with columns from the input data and an additional 'anomaly' column.
        If return_sql is True, returns the SQL query string instead.
        Returns ibis.Expr if input was ibis.Expr, pandas.DataFrame if input was pandas.DataFrame.

    Notes:
        - pandas is not required as a dependency - ibis.memtable() can handle pandas DataFrames if pandas is available.
        - Entity-Level Anomalies are detected for individual entities based on their own historical patterns, without considering the group context.
        - Group-Level Anomalies are detected for entities when compared to the behavior of other entities within the same group.
        - The function assumes that the input data has 'prediction' and 'resid' columns, which are typically obtained from a decomposition process.
        - GEH scores are used to measure the difference between predicted and actual values, taking into account the magnitude of the values.
        - MAD is used to detect anomalies based on the median absolute deviation of residuals within each group.
        - log_adjust_negative is used to make negative residuals more extreme when the data is censored at 0.
        - The function supports both pandas DataFrame and Ibis table expressions as input.
        - Connectivity analysis (originated_anomaly calculation) is only supported when entity_grouping_columns contains exactly one column.
    """
    # Validate parameter types
    assert isinstance(entity_grouping_columns, list), 'entity_grouping_columns must be a list.'
    assert group_grouping_columns is None or isinstance(group_grouping_columns, list), 'group_grouping_columns must be a list.'
    
    # Early validation: if connectivity_table is provided, ensure only one entity grouping column
    if connectivity_table is not None and len(entity_grouping_columns) != 1:
        raise ValueError("Connectivity analysis is currently only supported for a single entity grouping column.")

    # Check if data is an Ibis table
    if isinstance(decomposed_data, ibis.Expr):
        table = decomposed_data
    else:
        try:
            table = ibis.memtable(decomposed_data)
        except Exception as e:
            raise ValueError('Invalid data type. Please provide a valid Ibis table or pandas DataFrame.')

    # Validate required columns exist
    _validate_columns(table, datetime_column, value_column, entity_grouping_columns)
    
    # Assert that prediction and resid columns exist in the table
    assert 'prediction' in table.columns, 'prediction column not found in the table.'
    assert 'resid' in table.columns, 'resid column not found in the table.'

    ##############################
    ### FUNCTIONS ###
    ##############################
    epsilon = 1e-8

    # For making negative residuals more extreme
    def multiplier_func(value_column, prediction):
        return ibis.greatest(
            ((-1 * (value_column / (prediction + epsilon) + 0.1).log() + 2) / 2),
            ibis.literal(1)
        )
    
    def GEH_func(prediction, value_column):
        difference = prediction - value_column
        squared_diff = difference.pow(2)
        denominator = prediction + value_column + epsilon
        GEH = (2 * squared_diff / denominator).sqrt()
        signed_GEH = difference.sign() * GEH
        return signed_GEH
    
    def zscore_func(resid):
        return ((resid - resid.mean()) / (resid.std() + epsilon)).abs()
    
    def MAD_func(resid):
        return resid / (2 * resid.abs().median() + epsilon).abs()
    
    ##############################
    ### Entity Level Anomalies ###
    ##############################
    if GEH:
        # Transform residuals to GEH scores
        result = table.mutate(resid=GEH_func(table.prediction, table[value_column]))

        if log_adjust_negative:
            # Adjust negative GEH to be more extreme
            result = result.mutate(resid=_.resid * multiplier_func(table[value_column], table.prediction))
        # Handle NULL values in entity-level anomaly detection with GEH
        result = result.mutate(anomaly=(_.resid.abs() > entity_threshold).fill_null(False))
        
    else:
        if log_adjust_negative:
            # Adjust negative resid to be more extreme
            table = table.mutate(resid=_.resid * multiplier_func(table[value_column], table.prediction))
        # Handle NULL values in entity-level anomaly detection with z-scores
        result = (
            table
            .group_by(entity_grouping_columns)
            .mutate(anomaly=(zscore_func(_.resid) > entity_threshold).fill_null(False))
        )

    ##############################
    ### Group Level Anomalies ###
    ##############################
    if group_grouping_columns is not None:
        if MAD:
            result = (
                result
                .group_by(group_grouping_columns + [datetime_column])
                .mutate(anomaly=((MAD_func(_.resid) > group_threshold) & _.anomaly).fill_null(False))
            )
        else:
            result = (
                result
                .group_by(group_grouping_columns + [datetime_column])
                .mutate(anomaly=((zscore_func(_.resid) > group_threshold) & _.anomaly).fill_null(False))
            )

    if connectivity_table is not None:
        if isinstance(connectivity_table, ibis.Expr):
            conn_table = connectivity_table
        else:
            try:
                conn_table = ibis.memtable(connectivity_table)
            except Exception as e:
                raise ValueError('Invalid connectivity_table type. Please provide a valid Ibis table or pandas DataFrame.')

        # Validate connectivity table columns
        entity_col = entity_grouping_columns[0]
        
        # Find the next_entity_col in a case-insensitive way
        next_entity_col_name = None
        for col in conn_table.columns:
            if col.lower() == f"next_{entity_col}".lower():
                next_entity_col_name = col
                break
        
        if entity_col not in conn_table.columns or next_entity_col_name is None:
            raise ValueError(f"Connectivity table must contain '{entity_col}' and a 'next_{entity_col}' column (case-insensitive).")

        # Calculate originated anomaly
        anomaly_source = result.select(datetime_column, entity_col, 'anomaly')

        # Join with connectivity
        with_conn = anomaly_source.join(conn_table, entity_col)

        # Prepare next anomaly info
        next_anomaly_info = anomaly_source.rename(
            **{next_entity_col_name: entity_col}
        ).select(
            datetime_column,
            next_entity_col_name,
            next_anomaly='anomaly'
        )


        # Join to get next_anomaly
        merged = with_conn.join(
            next_anomaly_info,
            [datetime_column, next_entity_col_name]
        )

        # Cast to int for aggregation
        merged = merged.mutate(next_anomaly=_.next_anomaly.cast('int8'))

        # Get max_next_anomaly
        max_next_anomaly = merged.group_by(
            [datetime_column, entity_col]
        ).agg(
            max_next_anomaly=_.next_anomaly.max()
        )

        # Join back to the main result table (left join to preserve all rows)
        result = result.left_join(
            max_next_anomaly,
            [datetime_column, entity_col]
        ).select(result, 'max_next_anomaly')

        # Fill NA for entities that have no connectivity data
        # Use 1 so that anomaly > max_next_anomaly becomes False (not originated)
        result = result.mutate(max_next_anomaly=_.max_next_anomaly.fill_null(1))

        # Calculate originated_anomaly
        # Fill NULL anomalies with False before casting to ensure no NULL originated_anomaly values
        result = result.mutate(
            originated_anomaly=(_.anomaly.fill_null(False).cast('int8') > _.max_next_anomaly)
        ).drop('max_next_anomaly')

    if return_sql:
        return ibis.to_sql(result, dialect=dialect)
    elif isinstance(decomposed_data, ibis.Expr):
        return result  # Return Ibis expression directly if input was Ibis
    else:
        return result.execute()  # Convert to pandas for non-Ibis input