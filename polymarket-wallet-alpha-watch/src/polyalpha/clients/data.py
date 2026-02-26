from __future__ import annotations

from pydantic import BaseModel


class TradeRecord(BaseModel):
    proxy_wallet: str | None = None
    condition_id: str | None = None


class DataClient:
    """Data API client placeholder (implemented in Step B)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
