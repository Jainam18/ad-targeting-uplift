import numpy as np
import pandas as pd
import pytest

from uplift.config import COLUMNS, FEATURES, FLAGS
from uplift.data import csv_to_parquet, load_processed, validate_parquet


def _fake_csv(tmp_path, n=50, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    for c in FLAGS:
        df[c] = rng.integers(0, 2, size=n)
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return df, path


def test_roundtrip_preserves_rows_order_and_dtypes(tmp_path):
    df, csv = _fake_csv(tmp_path)
    pq = csv_to_parquet(csv, tmp_path / "out.parquet")

    report = validate_parquet(pq, expected_rows=len(df))
    out = load_processed(parquet_path=pq)

    assert report["n_rows"] == len(df)
    assert list(out.columns) == list(COLUMNS)
    assert all(out[c].dtype == np.float32 for c in FEATURES)
    assert all(out[c].dtype == np.int8 for c in FLAGS)
    np.testing.assert_allclose(out["f0"].to_numpy(), df["f0"].to_numpy(), rtol=1e-6)
    np.testing.assert_array_equal(out["treatment"].to_numpy(), df["treatment"].to_numpy())


def test_load_processed_selects_columns(tmp_path):
    _, csv = _fake_csv(tmp_path)
    pq = csv_to_parquet(csv, tmp_path / "out.parquet")
    assert list(load_processed(["treatment", "f3"], parquet_path=pq).columns) == ["treatment", "f3"]


def test_validation_rejects_wrong_row_count(tmp_path):
    df, csv = _fake_csv(tmp_path)
    pq = csv_to_parquet(csv, tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="n_rows"):
        validate_parquet(pq, expected_rows=len(df) + 1)


def test_validation_rejects_non_binary_flag(tmp_path):
    df, _ = _fake_csv(tmp_path)
    df.loc[0, "visit"] = 2
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)
    pq = csv_to_parquet(csv, tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="non_binary_flags"):
        validate_parquet(pq, expected_rows=len(df))
