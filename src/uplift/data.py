"""Raw CSV -> typed Parquet, validation, and loading."""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from uplift.config import COLUMNS, EXPECTED_ROWS, FEATURES, FLAGS, PARQUET, RAW_CSV


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def csv_to_parquet(csv_path: Path = RAW_CSV, parquet_path: Path = PARQUET) -> Path:
    """Stream the CSV into Parquet with float32 features and int8 flags, preserving row order."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_types = (
        ", ".join(f"'{c}': 'DOUBLE'" for c in FEATURES)
        + ", "
        + ", ".join(f"'{c}': 'BIGINT'" for c in FLAGS)
    )
    select = ", ".join(
        [f"CAST({c} AS FLOAT) AS {c}" for c in FEATURES]
        + [f"CAST({c} AS TINYINT) AS {c}" for c in FLAGS]
    )
    con = duckdb.connect()
    try:
        con.execute("SET preserve_insertion_order = true")
        con.execute(
            f"""
            COPY (
                SELECT {select}
                FROM read_csv('{_sql_path(csv_path)}', header = true, columns = {{{csv_types}}})
            ) TO '{_sql_path(parquet_path)}' (FORMAT parquet, COMPRESSION zstd)
            """
        )
    finally:
        con.close()
    return parquet_path


def validate_parquet(
    parquet_path: Path = PARQUET, expected_rows: int | None = EXPECTED_ROWS
) -> dict[str, Any]:
    """Structural checks only (no outcome-by-arm comparisons). Raises ValueError on failure."""
    con = duckdb.connect()
    try:
        src = f"read_parquet('{_sql_path(parquet_path)}')"
        schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
        columns = [row[0] for row in schema]
        dtypes = {row[0]: row[1] for row in schema}

        aggs = ["count(*) AS n_rows"]
        aggs += [f"count(*) - count({c}) AS nulls_{c}" for c in COLUMNS]
        aggs += [f"sum(CASE WHEN {c} NOT IN (0, 1) THEN 1 ELSE 0 END) AS bad_{c}" for c in FLAGS]
        aggs += [f"sum(CASE WHEN isfinite({c}) THEN 0 ELSE 1 END) AS inf_{c}" for c in FEATURES]
        row = con.execute(f"SELECT {', '.join(aggs)} FROM {src}").fetchdf().iloc[0]
    finally:
        con.close()

    report = {
        "n_rows": int(row["n_rows"]),
        "columns": columns,
        "dtypes": dtypes,
        "nulls": {c: int(row[f"nulls_{c}"]) for c in COLUMNS},
        "non_binary_flags": {c: int(row[f"bad_{c}"]) for c in FLAGS},
        "non_finite_features": {c: int(row[f"inf_{c}"]) for c in FEATURES},
        "file_size_mb": round(parquet_path.stat().st_size / 1e6, 1),
    }

    problems = []
    if columns != list(COLUMNS):
        problems.append(f"columns {columns} != expected {list(COLUMNS)}")
    if expected_rows is not None and report["n_rows"] != expected_rows:
        problems.append(f"n_rows {report['n_rows']} != expected {expected_rows}")
    for key in ("nulls", "non_binary_flags", "non_finite_features"):
        bad = {c: v for c, v in report[key].items() if v}
        if bad:
            problems.append(f"{key}: {bad}")
    if problems:
        raise ValueError("Parquet validation failed: " + "; ".join(problems))
    return report


def load_processed(
    columns: list[str] | tuple[str, ...] | None = None, parquet_path: Path = PARQUET
) -> pd.DataFrame:
    return pd.read_parquet(parquet_path, columns=list(columns) if columns else None)


if __name__ == "__main__":
    from uplift.io import write_results

    csv_to_parquet()
    write_results("data_prep", validate_parquet())
    print(f"Wrote {PARQUET}")
