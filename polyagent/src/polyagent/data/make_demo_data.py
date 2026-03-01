from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(
    out_path: Path,
    *,
    n_steps: int = 2000,
    seed: int = 7,
    step_ms: int = 1000,
) -> Path:
    rng = np.random.default_rng(seed)
    ts = np.arange(n_steps, dtype=np.int64) * step_ms

    latent = np.cumsum(rng.normal(0.0, 0.001, size=n_steps))
    base = 0.5 + latent
    base = np.clip(base, 0.05, 0.95)

    spread = np.clip(0.02 + rng.normal(0, 0.002, size=n_steps), 0.01, 0.05)
    best_bid = np.clip(base - spread / 2, 0.0, 1.0)
    best_ask = np.clip(base + spread / 2, 0.0, 1.0)

    bid_size_1 = rng.uniform(100, 400, size=n_steps)
    ask_size_1 = rng.uniform(100, 400, size=n_steps)
    bid_size_2 = rng.uniform(80, 300, size=n_steps)
    ask_size_2 = rng.uniform(80, 300, size=n_steps)
    bid_size_3 = rng.uniform(50, 250, size=n_steps)
    ask_size_3 = rng.uniform(50, 250, size=n_steps)

    trade_side = rng.choice([-1, 0, 1], size=n_steps, p=[0.33, 0.34, 0.33])
    trade_price_noise = rng.normal(0, 0.003, size=n_steps)
    trade_price = np.where(
        trade_side > 0,
        best_ask,
        np.where(trade_side < 0, best_bid, (best_bid + best_ask) / 2),
    ) + trade_price_noise
    trade_price = np.clip(trade_price, 0.0, 1.0)
    trade_size = np.where(trade_side == 0, 0.0, rng.uniform(5, 35, size=n_steps))

    df = pd.DataFrame(
        {
            "ts": ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size_1": bid_size_1,
            "ask_size_1": ask_size_1,
            "bid_size_2": bid_size_2,
            "ask_size_2": ask_size_2,
            "bid_size_3": bid_size_3,
            "ask_size_3": ask_size_3,
            "trade_price": trade_price,
            "trade_size": trade_size,
            "trade_side": trade_side,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        if out_path.suffix != ".parquet":
            out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
    return out_path
