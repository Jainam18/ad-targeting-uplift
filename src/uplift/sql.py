"""Run the DuckDB queries in sql/ against the processed Parquet file."""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from uplift.config import PARQUET, SQL


def run_sql(name: str, parquet_path: Path = PARQUET, **params: Any) -> pd.DataFrame:
    query = (SQL / f"{name}.sql").read_text()
    return duckdb.execute(query, {"path": str(parquet_path), **params}).df()
