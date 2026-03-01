from __future__ import annotations

from pathlib import Path

import pandas as pd

from polyagent.data.schemas import REQUIRED_COLUMNS


def load_market_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return df
