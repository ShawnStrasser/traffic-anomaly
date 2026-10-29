# Changelog

## 3.0.0 (October 29, 2026)

Ibis is gone: `decompose`, `anomaly` and `changepoint` are now written in plain DuckDB SQL. Results are identical to 2.5.4, checked value for value (including column types) on the bundled sample data and on 35 days of real 15-minute detector counts, under both DuckDB 1.5 and 2.0.

### Breaking changes

- **Ibis tables are no longer accepted.** Inputs can be a pandas DataFrame (or anything DuckDB can read, such as an Arrow table), a DuckDB relation, or the name of a table or view in a DuckDB connection.
- **Outputs follow the input.** A DataFrame comes back as a DataFrame, as before. A relation or a table name comes back as a lazy DuckDB relation in the same connection (previously an Ibis expression).
- **`return_sql=True` needs a table name.** The returned DuckDB SQL reads the input (and `connectivity_table`) by the names you passed. A DataFrame or relation with `return_sql=True` raises `ValueError`.
- **`dialect` is removed.** The SQL is DuckDB SQL only.
- **New `conn` argument** on all three functions: the DuckDB connection holding the table named by `data`. Defaults to DuckDB's default connection (the one `duckdb.sql()` uses).
- A string passed as `data` is read as a table name, so a name DuckDB cannot find raises `ValueError("Could not read table or view ...")`, and `data=None` raises the invalid data type error instead of a missing columns error.
- Requires Python 3.11+ and DuckDB 2.0+. Dependencies are now just `duckdb` and `pandas`; `ibis-framework` is no longer installed.

### Migrating from 2.x

| 2.x | 3.0 |
|-----|-----|
| `decompose(ibis_table, ...)` | `decompose(duckdb_relation, ...)` or `decompose('table_name', conn=con, ...)` |
| `decompose(df, ..., return_sql=True, dialect='duckdb')` | `con.register('my_table', df)`, then `decompose('my_table', conn=con, ..., return_sql=True)` |
| `anomaly(..., connectivity_table=ibis_table)` | a DataFrame, a relation (with relation input) or a table name (with table-name input) |
| `result.execute()` on an Ibis expression | `relation.df()` |

pandas-only code needs no changes.

### Other changes

- Intermediate columns use internal names, so input columns that happen to be called `detrend`, `time_of_day` and so on pass through instead of being overwritten.
- `changepoint` output is sorted by entity and time in all cases (it was unsorted when `recent_days_for_validation` was set).
- Tested on Python 3.11 to 3.15.
