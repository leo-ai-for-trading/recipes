from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from polyalpha.clients.base import BaseApiClient


class DataBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TradeRecord(DataBaseModel):
    proxy_wallet: str | None = Field(
        default=None,
        validation_alias=AliasChoices("proxyWallet", "proxy_wallet", "user"),
    )
    condition_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("conditionId", "condition_id", "market"),
    )
    event_id: str | None = Field(default=None, validation_alias=AliasChoices("eventId", "event_id"))
    side: str | None = None
    price: float | None = None
    size: float | None = None
    title: str | None = Field(default=None, validation_alias=AliasChoices("title", "question"))
    timestamp: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "timestamp",
            "timeStamp",
            "createdAt",
            "created_at",
            "updatedAt",
            "updated_at",
            "tradedAt",
            "traded_at",
        ),
    )


class PositionRecord(DataBaseModel):
    condition_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("conditionId", "condition_id", "market"),
    )
    question: str | None = Field(default=None, validation_alias=AliasChoices("question", "title"))
    outcome: str | None = None
    total_bought: float | None = Field(
        default=None,
        validation_alias=AliasChoices("totalBought", "total_bought"),
    )
    realized_pnl: float | None = Field(
        default=None,
        validation_alias=AliasChoices("realizedPnl", "realized_pnl"),
    )
    cash_pnl: float | None = Field(default=None, validation_alias=AliasChoices("cashPnl", "cash_pnl", "pnl"))
    current_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("currentValue", "current_value"),
    )
    initial_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("initialValue", "initial_value"),
    )
    opened_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("openedAt", "opened_at", "createdAt", "created_at"),
    )
    closed_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("closedAt", "closed_at", "updatedAt", "updated_at"),
    )

    def resolved_condition_id(self) -> str | None:
        return self.condition_id


class DataClient(BaseApiClient):
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

    async def list_trades(
        self,
        *,
        condition_ids: list[str] | None = None,
        user: str | None = None,
        event_id: str | None = None,
        side: str | None = None,
        page_size: int = 500,
    ) -> list[TradeRecord]:
        params: dict[str, str] = {}
        if condition_ids:
            params["market"] = self._join_csv(condition_ids)
        if user:
            params["user"] = user
        if event_id:
            params["eventId"] = event_id
        if side:
            params["side"] = side

        items = await self._paginate("/trades", params=params, page_size=page_size)
        return [TradeRecord.model_validate(item) for item in items]

    async def list_positions(
        self,
        *,
        user: str,
        condition_ids: list[str] | None = None,
        page_size: int = 500,
    ) -> list[PositionRecord]:
        params: dict[str, str] = {"user": user}
        if condition_ids:
            params["market"] = self._join_csv(condition_ids)
        items = await self._paginate("/positions", params=params, page_size=page_size)
        return [PositionRecord.model_validate(item) for item in items]

    async def list_closed_positions(
        self,
        *,
        user: str,
        condition_ids: list[str] | None = None,
        page_size: int = 500,
    ) -> list[PositionRecord]:
        params: dict[str, str] = {"user": user}
        if condition_ids:
            params["market"] = self._join_csv(condition_ids)
        items = await self._paginate("/closed-positions", params=params, page_size=page_size)
        return [PositionRecord.model_validate(item) for item in items]

    @staticmethod
    def _join_csv(values: list[str]) -> str:
        return ",".join(v for v in values if v)
