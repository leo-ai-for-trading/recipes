from __future__ import annotations

from pydantic import BaseModel, Field


class MarketRow(BaseModel):
    ts: int
    best_bid: float
    best_ask: float
    bid_size_1: float
    ask_size_1: float
    bid_size_2: float
    ask_size_2: float
    bid_size_3: float
    ask_size_3: float
    trade_price: float
    trade_size: float
    trade_side: int = Field(description="-1 sell, 0 none, +1 buy")


REQUIRED_COLUMNS = [
    "ts",
    "best_bid",
    "best_ask",
    "bid_size_1",
    "ask_size_1",
    "bid_size_2",
    "ask_size_2",
    "bid_size_3",
    "ask_size_3",
    "trade_price",
    "trade_size",
    "trade_side",
]
