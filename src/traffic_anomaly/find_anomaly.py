import ibis
from ibis import _

def find_anomaly(decomposed_data, threshold=3.0, MAD=False, return_sql=False):
    """
    Detect anomalies in the decomposed time series data.

    Args:
        decomposed_data (pd.DataFrame or ibis.expr.types.TableExpr): The time series data to decompose.
        threshold (float): The threshold z-score for anomaly detection (default: 3.0).

    Returns:
        pandas.DataFrame: The detected anomalies. If return_sql is True, returns the SQL query string instead.

    
    NOTES: Entity-Level Anomalies are detected for individual entities based on their own historical patterns, without considering the group context.
    Group-Level Anomalies are detected for entities when compared to the behavior of other entities within the same group.
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
    assert 'id' in table.columns, 'id column not found in the table.'
    assert 'resid' in table.columns, 'resid column not found in the table.'

    epsilon = 1e-8
    
    if MAD:
        # Calculate the Median Absolute Deviation (MAD)
        result = (
            table
            .group_by('id').mutate(MAD=_.resid.abs().median())
            .mutate(anomaly=_.resid / (2 * _.MAD + epsilon) > threshold)
            .drop('MAD')
            )
        
    else:
        result = (
            table
            .group_by('id')
            .mutate(anomaly=(_.resid - _.resid.mean()) / (_.resid.std() + epsilon) > threshold)
        )

    # Check if `group` column exists in the table
    if 'group' in result.columns:
        # Check that timestamp column exists in the table
        assert 'timestamp' in result.columns, 'timestamp column not found in the table.'
        if MAD:
            result = (
                result
                .group_by(['timestamp', 'group']).mutate(MAD=_.resid.abs().median())
                .mutate(anomaly=(_.resid / (2 * _.MAD + epsilon) > threshold) & _.anomaly)
                .drop('MAD')
            )
        else:
            result = (
                result
                .group_by(['timestamp', 'group'])
                .mutate(anomaly=(
                    ((_.resid - _.resid.mean()) / (_.resid.std() + epsilon) > threshold) &
                    (_.anomaly)
                ))
            )
    
    result = result.order_by([_.timestamp, _.id])

    if return_sql:
        return ibis.to_sql(result)
    else:
        return result.execute()