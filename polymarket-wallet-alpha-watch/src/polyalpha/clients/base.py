from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ApiClientError(RuntimeError):
    """Raised when an API request fails after retries."""


class AsyncRateLimiter:
    """Very small token-interval limiter for request pacing."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            current = time.monotonic()
            self._next_allowed = max(self._next_allowed, current) + self._interval


class BaseApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        max_concurrency: int,
        requests_per_second: float,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers=default_headers,
        )
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_limiter = AsyncRateLimiter(requests_per_second=requests_per_second)

    async def __aenter__(self) -> BaseApiClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            response: httpx.Response | None = None
            try:
                await self._rate_limiter.acquire()
                async with self._semaphore:
                    response = await self._client.request(method, path, params=params)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await self._sleep_before_retry(attempt)
                continue

            if response.status_code in retryable_statuses:
                last_error = ApiClientError(
                    f"{method} {path} returned retryable status {response.status_code}"
                )
                if attempt >= self._max_retries:
                    break
                await self._sleep_before_retry(attempt, response=response)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = (
                    f"{method} {path} failed with status {response.status_code}: "
                    f"{response.text[:300]}"
                )
                raise ApiClientError(message) from exc

            if not response.content:
                return {}

            try:
                return response.json()
            except ValueError as exc:
                raise ApiClientError(f"{method} {path} returned non-JSON response") from exc

        raise ApiClientError(
            f"{method} {path} failed after retries"
            + (f": {last_error}" if last_error else "")
        )

    async def _sleep_before_retry(
        self,
        attempt: int,
        *,
        response: httpx.Response | None = None,
    ) -> None:
        retry_after: float | None = None
        if response is not None and "Retry-After" in response.headers:
            raw = response.headers["Retry-After"]
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None

        base = self._backoff_seconds * (2**attempt)
        jitter = random.uniform(0.0, self._backoff_seconds)
        delay = retry_after if retry_after is not None else base + jitter
        logger.debug("Retrying request after %.3fs", delay)
        await asyncio.sleep(delay)

    async def _paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        page_size: int = 500,
        limit_param: str = "limit",
        offset_param: str = "offset",
        data_key: str | None = None,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        offset = 0

        while True:
            page_params = dict(params or {})
            page_params[limit_param] = page_size
            page_params[offset_param] = offset

            payload = await self._request_json("GET", path, params=page_params)
            items = self._extract_items(payload, data_key=data_key)
            if not items:
                break

            collected.extend(items)

            if len(items) < page_size:
                break
            offset += len(items)

        return collected

    @staticmethod
    def _extract_items(payload: Any, *, data_key: str | None) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            if data_key and isinstance(payload.get(data_key), list):
                data = payload.get(data_key, [])
                return [item for item in data if isinstance(item, dict)]

            for key in ("data", "items", "results"):
                maybe_items = payload.get(key)
                if isinstance(maybe_items, list):
                    return [item for item in maybe_items if isinstance(item, dict)]

        return []
