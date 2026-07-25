from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import json

import pandas as pd


def read_tabular(locator: str | Path) -> pd.DataFrame:
    text = str(locator)
    suffix = Path(urlparse(text).path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(text)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(text, lines=suffix == ".jsonl")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(text)
    return pd.read_csv(text)


def write_frame(frame: pd.DataFrame, path_without_suffix: str | Path) -> Path:
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet, index=False)
        return parquet
    except (ImportError, ModuleNotFoundError, ValueError):
        csv_path = base.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def write_json(data: object, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target
