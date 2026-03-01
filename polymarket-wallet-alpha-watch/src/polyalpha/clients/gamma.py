from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from polyalpha.clients.base import BaseApiClient


class GammaBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class GammaTag(GammaBaseModel):
    id: int
    slug: str | None = None
    label: str | None = Field(default=None, validation_alias=AliasChoices("label", "name"))


class GammaMarket(GammaBaseModel):
    condition_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("conditionId", "condition_id"),
    )
    question: str | None = Field(default=None, validation_alias=AliasChoices("question", "title"))
    active: bool | None = None
    closed: bool | None = None
    end_date: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("endDate", "end_date"),
    )
    tag_ids: list[int] = Field(default_factory=list)
    tag_slugs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_tag_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        copied = dict(value)
        raw_tag_ids = copied.get("tagIds") or copied.get("tag_ids") or []
        if isinstance(raw_tag_ids, list):
            copied["tag_ids"] = [int(item) for item in raw_tag_ids if str(item).isdigit()]

        tag_slugs: list[str] = []
        raw_tags = copied.get("tags")
        if isinstance(raw_tags, list):
            for raw_tag in raw_tags:
                if not isinstance(raw_tag, dict):
                    continue
                raw_id = raw_tag.get("id")
                if raw_id is not None and str(raw_id).isdigit():
                    copied.setdefault("tag_ids", []).append(int(raw_id))
                raw_slug = raw_tag.get("slug")
                if isinstance(raw_slug, str):
                    tag_slugs.append(raw_slug)
        copied["tag_slugs"] = tag_slugs
        copied["tag_ids"] = sorted(set(copied.get("tag_ids", [])))
        return copied


class GammaPublicProfile(GammaBaseModel):
    address: str | None = None
    name: str | None = None
    pseudonym: str | None = Field(default=None, validation_alias=AliasChoices("pseudonym", "displayName"))
    x_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("xUsername", "x_username", "twitterUsername"),
    )
    verified_badge: bool = Field(
        default=False,
        validation_alias=AliasChoices("verifiedBadge", "verified_badge"),
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        copied = dict(value)
        raw_tags = copied.get("tags")
        if isinstance(raw_tags, list):
            tags: list[str] = []
            for raw_tag in raw_tags:
                if isinstance(raw_tag, str):
                    tags.append(raw_tag)
                elif isinstance(raw_tag, dict):
                    label = raw_tag.get("label") or raw_tag.get("name")
                    if isinstance(label, str):
                        tags.append(label)
            copied["tags"] = tags
        return copied


class GammaClient(BaseApiClient):
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

    async def list_tags(self) -> list[GammaTag]:
        items = await self._paginate("/tags", page_size=500)
        return [GammaTag.model_validate(item) for item in items]

    async def list_markets_by_tag(
        self,
        *,
        tag_id: int,
        include_closed_markets: bool,
    ) -> list[GammaMarket]:
        params = {
            "tag_id": tag_id,
            "closed": str(include_closed_markets).lower(),
        }
        items = await self._paginate("/markets", params=params, page_size=500)
        return [GammaMarket.model_validate(item) for item in items]

    async def public_search(self, query: str) -> list[dict[str, Any]]:
        if not query:
            return []
        items = await self._paginate("/public-search", params={"q": query}, page_size=100)
        return items

    async def get_public_profile(self, address: str) -> GammaPublicProfile | None:
        payload = await self._request_json("GET", "/public-profile", params={"address": address})
        if payload in (None, {}, []):
            return None

        if isinstance(payload, dict) and "profile" in payload and isinstance(payload["profile"], dict):
            return GammaPublicProfile.model_validate(payload["profile"])

        if isinstance(payload, dict):
            return GammaPublicProfile.model_validate(payload)

        return None
