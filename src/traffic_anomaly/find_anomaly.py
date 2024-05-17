import ibis
from ibis import _

def find_anomaly(
        decomposed_data,
        datetime_column,
        value_column,
        entity_grouping_columns, #=['id'],
        group_grouping_columns=None, #['group'],
        entity_threshold=3.5,
        group_threshold=3.5,
        GEH=False, # only impacts entity level
        MAD=False, # only impacts group level
        log_adjust_negative=False, # makes negative residuals more extreme, for when data is censored at 0
        return_sql=False,
        ):
    """
    Detect anomalies in the time series data at both entity and group levels.

    Args:
        decomposed_data (pd.DataFrame or ibis.expr.types.TableExpr): The decomposed time series data.
        datetime_column (str): The name of the column containing datetime information.
        value_column (str): The name of the column containing the actual values.
        entity_grouping_columns (list): The list of columns used for grouping entities (default: ['id']).
        group_grouping_columns (list, optional): The list of columns used for grouping groups (default: ['group']).
        entity_threshold (float): The threshold for detecting anomalies at the entity level (default: 3.5).
        group_threshold (float): The threshold for detecting anomalies at the group level (default: 3.5).
        GEH (bool): Whether to use GEH scores for entity-level anomaly detection (default: False).
        MAD (bool): Whether to use Median Absolute Deviation (MAD) for group-level anomaly detection (default: False).
        log_adjust_negative (bool): Whether to make negative residuals more extreme for data censored at 0 (default: False).
        return_sql (bool): Whether to return the SQL query string instead of the result (default: False).

    Returns:
        pandas.DataFrame: The detected anomalies with columns from the input data and an additional 'anomaly' column.
        If return_sql is True, returns the SQL query string instead.

    Notes:
        - Entity-Level Anomalies are detected for individual entities based on their own historical patterns, without considering the group context.
        - Group-Level Anomalies are detected for entities when compared to the behavior of other entities within the same group.
        - The function assumes that the input data has 'prediction' and 'resid' columns, which are typically obtained from a decomposition process.
        - GEH scores are used to measure the difference between predicted and actual values, taking into account the magnitude of the values.
        - MAD is used to detect anomalies based on the median absolute deviation of residuals within each group.
        - log_adjust_negative is used to make negative residuals more extreme when the data is censored at 0.
        - The function supports both pandas DataFrame and Ibis table expressions as input.
    """
    # Check if df_or_table is an Ibis table
    if isinstance(decomposed_data, ibis.Expr):
        table = decomposed_data
    else:
        try:
            table = ibis.memtable(decomposed_data)
        except Exception as e:
            raise ValueError('Invalid data type. Please provide a valid Ibis table or Pandas DataFrame.')

    # Assert that id and resid columns exist in the table
    assert 'prediction' in table.columns, 'prediction column not found in the table.'
    assert 'resid' in table.columns, 'resid column not found in the table.'
    # Assert that the entity_grouping_columns is a list
    assert isinstance(entity_grouping_columns, list), 'entity_grouping_columns must be a list.'
    # Assert that the group_grouping_columns is a list if not None
    if group_grouping_columns is not None:
        assert isinstance(group_grouping_columns, list), 'group_grouping_columns must be a list.'


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
        result = result.mutate(anomaly=_.resid.abs() > entity_threshold)
        
    else:
        if log_adjust_negative:
            # Adjust negative resid to be more extreme
            table = table.mutate(resid=_.resid * multiplier_func(table[value_column], table.prediction))
        result = (
            table
            .group_by(entity_grouping_columns)
            .mutate(anomaly=zscore_func(_.resid) > entity_threshold)
        )

    ##############################
    ### Group Level Anomalies ###
    ##############################
    if group_grouping_columns is not None:
        if MAD:
            result = (
                result
                .group_by(group_grouping_columns + [datetime_column])
                .mutate(anomaly=(MAD_func(_.resid) > group_threshold) & _.anomaly)
            )
        else:
            result = (
                result
                .group_by(group_grouping_columns + [datetime_column])
                .mutate(anomaly=(zscore_func(_.resid) > group_threshold) & _.anomaly)
            )
        #result = result.drop(group_grouping_columns)

    result = result.drop('resid')
    #result = result.order_by(datetime_column)

    if return_sql:
        return ibis.to_sql(result)
    else:
        return result.execute()