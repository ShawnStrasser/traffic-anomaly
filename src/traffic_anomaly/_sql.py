"""Shared input handling for the DuckDB queries behind decompose, anomaly and changepoint."""
import itertools
import numbers
from typing import List

import duckdb

# Names the queries read their inputs from when the data is not a table in the caller's connection
INPUT = 'traffic_anomaly_input'
CONNECTIVITY = 'traffic_anomaly_connectivity'
# DuckDB 1.x cannot nest relation queries that use the same view name, so each call on a relation gets its own
_relation_numbers = itertools.count()


def quote(name: str) -> str:
    """A column name as a quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def columns(names: List[str], alias: str = '') -> str:
    """Comma-separated quoted column names, optionally qualified by a table alias."""
    prefix = f'{alias}.' if alias else ''
    return ', '.join(prefix + quote(name) for name in names)


def number(value) -> str:
    """A numeric parameter as a SQL literal."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'Expected a number, got {value!r}')
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return repr(float(value))


def interval(amount, unit: str) -> str:
    """A SQL interval literal, e.g. INTERVAL '7' DAY."""
    return f"INTERVAL '{number(amount)}' {unit}"


def _query_relation(relation, name, sql, traffic_anomaly_connectivity=None):
    """Run sql on a DuckDB relation, which the query reads by name, returning a lazy relation.

    A connectivity table passed in is found by name through DuckDB's replacement scan, which looks up
    names missing from the catalog among the local variables of the function that runs the query.
    """
    return relation.query(name, sql)


class Source:
    """The input to one function call: where its query reads the data and how the result comes back.

    data is a pandas DataFrame (or anything else DuckDB can read, such as an Arrow table), a DuckDB
    relation, or the name of a table or view in conn (DuckDB's default connection if conn is None).
    A DataFrame is read in a private connection and comes back as a DataFrame. A relation or table name
    comes back as a lazy DuckDB relation, or with return_sql=True (table names only) as the SQL text.

    DataFrames are handled as in 2.x: timestamps are read in microseconds, time zones
    are worked in UTC, and time zone aware columns come back in the time zone they went in with.
    """

    def __init__(self, data, conn, return_sql, datetime_column, value_column, grouping_columns):
        self.data = data
        self.return_sql = return_sql
        self.connectivity = None
        self._con = None
        self._time_zones = {}
        if isinstance(data, str):
            self.conn = duckdb.default_connection() if conn is None else conn
            self.table = data
            schema = _describe(self.conn, data)
        elif return_sql:
            raise ValueError('return_sql=True needs data given as the name of a table or view.')
        elif isinstance(data, duckdb.DuckDBPyRelation):
            self.table = f'{INPUT}_{next(_relation_numbers)}'
            schema = data
        else:
            self.table = INPUT
            self._con = duckdb.connect()
            self._con.execute("SET TimeZone = 'UTC'")
            try:
                schema = _register_frame(self._con, INPUT, data)
            except Exception:
                self._con.close()
                if hasattr(data, 'columns') and len(data.columns) == 0:
                    _validate([], {}, datetime_column, value_column, grouping_columns)
                raise ValueError('Invalid data type. Please provide a pandas DataFrame, a DuckDB relation '
                                 'or the name of a table or view.') from None
            self._time_zones = {name: dtype.tz for name, dtype in getattr(data, 'dtypes', {}).items()
                                if getattr(dtype, 'tz', None) is not None}
        self.columns = list(schema.columns)
        self.types = {name: str(dtype) for name, dtype in zip(schema.columns, schema.types)}
        _validate(self.columns, self.types, datetime_column, value_column, grouping_columns)

    def add_connectivity(self, table):
        """Make the connectivity table readable by this call's query. Returns (SQL name, its columns)."""
        if isinstance(self.data, str):
            if not isinstance(table, str):
                raise ValueError('With data given as a table name, connectivity_table must be a table name '
                                 'in the same connection.')
            return table, list(_describe(self.conn, table).columns)
        if self._con is not None:
            try:
                return CONNECTIVITY, list(_register_frame(self._con, CONNECTIVITY, table).columns)
            except Exception:
                raise ValueError('Invalid connectivity_table type. Please provide a pandas DataFrame.') from None
        self.connectivity = table
        if isinstance(table, duckdb.DuckDBPyRelation):
            return CONNECTIVITY, list(table.columns)
        with duckdb.connect() as con:
            try:
                con.register(CONNECTIVITY, table)
            except Exception:
                raise ValueError('Invalid connectivity_table type. Please provide a pandas DataFrame or a '
                                 'DuckDB relation.') from None
            return CONNECTIVITY, list(con.table(CONNECTIVITY).columns)

    def result(self, sql):
        """The finished query as this kind of input expects it back."""
        if self.return_sql:
            return sql
        if isinstance(self.data, str):
            return self.conn.sql(sql)
        if self._con is None:
            return _query_relation(self.data, self.table, sql, self.connectivity)
        try:
            df = self._con.sql(sql).df()
        finally:
            self._con.close()
        for name, tz in self._time_zones.items():
            if name in df.columns and getattr(df[name].dtype, 'tz', None) is not None:
                df[name] = df[name].dt.tz_convert(tz)
        return df


def _register_frame(con, name, frame):
    """Expose a DataFrame to con as a view called name, with its timestamps in microseconds."""
    con.register(f'{name}_frame', frame)
    relation = con.table(f'{name}_frame')
    casts = [f'CAST({quote(col)} AS TIMESTAMP) AS {quote(col)}' for col, dtype in zip(relation.columns, relation.types)
             if str(dtype) in ('TIMESTAMP_NS', 'TIMESTAMP_MS', 'TIMESTAMP_S')]
    replace = f" REPLACE ({', '.join(casts)})" if casts else ''
    con.execute(f'CREATE VIEW {name} AS SELECT *{replace} FROM {name}_frame')
    return con.table(name)


def _describe(conn, table):
    """A relation over a table or view, used for its column names and types."""
    try:
        return conn.sql(f'SELECT * FROM {table} LIMIT 0')
    except duckdb.Error as e:
        raise ValueError(f"Could not read table or view '{table}': {e}") from None


def _validate(names, types, datetime_column, value_column, grouping_columns):
    """Check that the required columns exist and the datetime column is temporal."""
    missing = [col for col in [datetime_column, value_column] + list(grouping_columns) if col not in names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not types[datetime_column].startswith(('DATE', 'TIME')):
        raise TypeError(f"Column '{datetime_column}' must be a temporal type (e.g., timestamp or date), "
                        f"but is {types[datetime_column]}")
