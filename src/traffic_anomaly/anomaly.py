from typing import Union, List, Optional, Any

import duckdb

from ._sql import Source, quote, columns, number

EPSILON = '1e-08'


def anomaly(
    decomposed_data: Union[str, duckdb.DuckDBPyRelation, Any],  # table name, DuckDB relation or pandas DataFrame
    datetime_column: str,
    value_column: str,
    entity_grouping_columns: List[str],
    group_grouping_columns: Optional[List[str]] = None,
    entity_threshold: float = 3.5,
    group_threshold: float = 3.5,
    GEH: bool = False,
    MAD: bool = False,
    log_adjust_negative: bool = False,
    connectivity_table: Optional[Union[str, duckdb.DuckDBPyRelation, Any]] = None,
    return_sql: bool = False,
    conn: duckdb.DuckDBPyConnection = None
) -> Union[duckdb.DuckDBPyRelation, Any, str]:  # DuckDB relation, pandas DataFrame, or str
    """
    Detect anomalies in the time series data at both entity and group levels.

    Args:
        decomposed_data: The decomposed time series data. A pandas DataFrame (or anything else DuckDB can read),
                         a DuckDB relation, or the name of a table or view in conn.
        datetime_column: The name of the column containing datetime information.
        value_column: The name of the column containing the actual values.
        entity_grouping_columns: The list of columns used for grouping entities.
        group_grouping_columns: The list of columns used for grouping groups.
        entity_threshold: The threshold for detecting anomalies at the entity level.
        group_threshold: The threshold for detecting anomalies at the group level.
        GEH: Whether to use GEH scores for entity-level anomaly detection.
        MAD: Whether to use Median Absolute Deviation (MAD) for group-level anomaly detection.
        log_adjust_negative: Whether to make negative residuals more extreme for data censored at 0.
        connectivity_table: Optional table to calculate originated anomalies, given the same way as
                            decomposed_data (with a relation, a DataFrame also works).
                            When provided, only a single entity_grouping_column is supported.
        return_sql: Whether to return the DuckDB SQL query instead of the result. Requires decomposed_data
                    (and connectivity_table) to be table or view names; the query reads them by those names.
        conn: The DuckDB connection holding the tables or views named by decomposed_data and
              connectivity_table. Defaults to DuckDB's default connection. Not used for DataFrames or relations.

    Returns:
        The detected anomalies with columns from the input data and an additional 'anomaly' column.
        A pandas DataFrame if decomposed_data was a DataFrame, otherwise a lazy DuckDB relation in the same
        connection. If return_sql is True, returns the SQL query string instead.

    Notes:
        - Entity-Level Anomalies are detected for individual entities based on their own historical patterns, without considering the group context.
        - Group-Level Anomalies are detected for entities when compared to the behavior of other entities within the same group.
        - The function assumes that the input data has 'prediction' and 'resid' columns, which are typically obtained from a decomposition process.
        - GEH scores are used to measure the difference between predicted and actual values, taking into account the magnitude of the values.
        - MAD is used to detect anomalies based on the median absolute deviation of residuals within each group.
        - log_adjust_negative is used to make negative residuals more extreme when the data is censored at 0.
        - Connectivity analysis (originated_anomaly calculation) is only supported when entity_grouping_columns contains exactly one column.
    """
    # Validate parameter types
    assert isinstance(entity_grouping_columns, list), 'entity_grouping_columns must be a list.'
    assert group_grouping_columns is None or isinstance(group_grouping_columns, list), 'group_grouping_columns must be a list.'

    # Early validation: if connectivity_table is provided, ensure only one entity grouping column
    if connectivity_table is not None and len(entity_grouping_columns) != 1:
        raise ValueError("Connectivity analysis is currently only supported for a single entity grouping column.")

    source = Source(decomposed_data, conn, return_sql, datetime_column, value_column, entity_grouping_columns)

    # Assert that prediction and resid columns exist in the table
    assert 'prediction' in source.columns, 'prediction column not found in the table.'
    assert 'resid' in source.columns, 'resid column not found in the table.'

    dt, value = quote(datetime_column), quote(value_column)

    def with_anomaly(expr):
        """Select all columns with expr as the anomaly column, replacing it in place if the input has one."""
        if 'anomaly' in source.columns:
            return f'* REPLACE ({expr} AS anomaly)'
        return f'*, {expr} AS anomaly'

    ##############################
    ### FUNCTIONS ###
    ##############################
    # For making negative residuals more extreme
    multiplier = f"GREATEST(((LN(({value} / (prediction + {EPSILON})) + 0.1) * -1) + 2) / 2, 1)"
    geh = (f"SIGN(prediction - {value}) * "
           f"SQRT((POWER(prediction - {value}, 2) * 2) / (prediction + {value} + {EPSILON}))")

    def zscore(window):
        return f"ABS((resid - AVG(resid) OVER {window}) / (STDDEV_SAMP(resid) OVER {window} + {EPSILON}))"

    def mad(window):
        return f"resid / ABS((QUANTILE_CONT(ABS(resid), 0.5) OVER {window} * 2) + {EPSILON})"

    ##############################
    ### Entity Level Anomalies ###
    ##############################
    if GEH:
        # Transform residuals to GEH scores, adjusting negative GEH to be more extreme if asked
        resid = f"({geh}) * {multiplier}" if log_adjust_negative else geh
        scored = f"SELECT * REPLACE ({resid} AS resid) FROM {source.table}"
        entity = f"COALESCE(ABS(resid) > {number(entity_threshold)}, FALSE)"
    else:
        if log_adjust_negative:
            # Adjust negative resid to be more extreme
            scored = f"SELECT * REPLACE (resid * {multiplier} AS resid) FROM {source.table}"
        else:
            scored = f"SELECT * FROM {source.table}"
        window = f"(PARTITION BY {columns(entity_grouping_columns)})"
        entity = f"COALESCE({zscore(window)} > {number(entity_threshold)}, FALSE)"

    ctes = [f"scored AS ({scored})", f"entity_level AS (SELECT {with_anomaly(entity)} FROM scored)"]
    result = 'entity_level'

    ##############################
    ### Group Level Anomalies ###
    ##############################
    if group_grouping_columns is not None:
        window = f"(PARTITION BY {columns(group_grouping_columns + [datetime_column])})"
        score = mad(window) if MAD else zscore(window)
        group = f"COALESCE(({score} > {number(group_threshold)}) AND anomaly, FALSE)"
        ctes.append(f"group_level AS (SELECT * REPLACE ({group} AS anomaly) FROM entity_level)")
        result = 'group_level'

    if connectivity_table is None:
        sql = f"WITH {', '.join(ctes)} SELECT * FROM {result}"
        return source.result(sql)

    conn_table, conn_columns = source.add_connectivity(connectivity_table)

    # Validate connectivity table columns
    entity_col = entity_grouping_columns[0]

    # Find the next_entity_col in a case-insensitive way
    next_entity_col_name = None
    for col in conn_columns:
        if col.lower() == f"next_{entity_col}".lower():
            next_entity_col_name = col
            break

    if entity_col not in conn_columns or next_entity_col_name is None:
        raise ValueError(f"Connectivity table must contain '{entity_col}' and a 'next_{entity_col}' column (case-insensitive).")

    ent, next_ent = quote(entity_col), quote(next_entity_col_name)
    # An anomaly originated at an entity when no downstream entity had one at the same time. Entities
    # without connectivity data count as having a downstream anomaly, so they are never originated.
    ctes.append(f"""
    next_anomaly AS (
        SELECT a.{dt}, a.{ent}, MAX(CAST(n.anomaly AS TINYINT)) AS max_next_anomaly
        FROM {result} a
        JOIN {conn_table} c ON a.{ent} = c.{ent}
        JOIN {result} n ON a.{dt} = n.{dt} AND c.{next_ent} = n.{ent}
        GROUP BY a.{dt}, a.{ent}
    )""")
    originated = "CAST(COALESCE(r.anomaly, FALSE) AS TINYINT) > COALESCE(n.max_next_anomaly, 1)"
    if 'originated_anomaly' in source.columns:
        select = f"r.* REPLACE ({originated} AS originated_anomaly)"
    else:
        select = f"r.*, {originated} AS originated_anomaly"
    sql = f"""
    WITH {', '.join(ctes)}
    SELECT {select}
    FROM {result} r
    LEFT JOIN next_anomaly n ON r.{dt} = n.{dt} AND r.{ent} = n.{ent}
    """
    return source.result(sql)
