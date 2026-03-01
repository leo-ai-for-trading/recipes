from __future__ import annotations

from typing import Any

from polyalpha.clients.base import BaseApiClient


class ClobClient(BaseApiClient):
    """Optional CLOB access for mark-to-market enrichment."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        max_concurrency: int,
        requests_per_second: float,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            max_concurrency=max_concurrency,
            requests_per_second=requests_per_second,
        )

    async def get_book(self, token_id: str) -> dict[str, Any]:
        return await self._request_json("GET", "/book", params={"token_id": token_id})

    async def get_last_trade_price(self, token_id: str) -> float | None:
        payload = await self._request_json("GET", "/last-trade-price", params={"token_id": token_id})
        if isinstance(payload, dict):
            value = payload.get("price") or payload.get("last_price")
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
        return None
