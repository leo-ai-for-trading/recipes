from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from polyalpha.clients.gamma import GammaClient, GammaMarket
from polyalpha.config import TopicConfig

logger = logging.getLogger(__name__)

MAX_RELATED_TAGS = 50


@dataclass(slots=True)
class DiscoveredMarket:
    condition_id: str
    question: str | None
    tag_ids: list[int] = field(default_factory=list)
    tag_slugs: list[str] = field(default_factory=list)
    closed: bool | None = None


@dataclass(slots=True)
class DiscoveryResult:
    topic_name: str
    tag_ids: list[int]
    condition_ids: list[str]
    markets: dict[str, DiscoveredMarket]


async def run_discover_markets(
    *,
    topic_name: str,
    topic: TopicConfig,
    gamma_client: GammaClient,
) -> DiscoveryResult:
    tags = await gamma_client.list_tags()
    slug_to_id = {tag.slug: tag.id for tag in tags if tag.slug}

    seed_tag_ids = set(topic.tag_ids)
    missing_slugs: list[str] = []
    for slug in topic.tag_slugs:
        tag_id = slug_to_id.get(slug)
        if tag_id is None:
            missing_slugs.append(slug)
            continue
        seed_tag_ids.add(tag_id)

    search_tag_ids: set[int] = set()
    search_condition_ids: set[str] = set()
    if missing_slugs or not seed_tag_ids:
        for keyword in topic.keywords:
            search_rows = await gamma_client.public_search(keyword)
            for row in search_rows:
                search_tag_ids.update(extract_tag_ids_from_search_row(row))
                search_condition_ids.update(extract_condition_ids_from_search_row(row))

    all_seed_tag_ids = sorted(seed_tag_ids | search_tag_ids)
    discovered_markets: dict[str, DiscoveredMarket] = {}

    first_pass_markets = await _discover_markets_for_tags(
        gamma_client=gamma_client,
        tag_ids=all_seed_tag_ids,
        include_closed_markets=topic.include_closed_markets,
    )
    _merge_discovered_markets(discovered_markets, first_pass_markets)

    if topic.include_related_tags:
        related_tag_ids = set()
        for market in first_pass_markets:
            related_tag_ids.update(market.tag_ids)
        related_tag_ids.difference_update(all_seed_tag_ids)

        related_subset = sorted(related_tag_ids)[:MAX_RELATED_TAGS]
        if related_subset:
            logger.info(
                "Expanding topic '%s' with %d related tags",
                topic_name,
                len(related_subset),
            )
            related_markets = await _discover_markets_for_tags(
                gamma_client=gamma_client,
                tag_ids=related_subset,
                include_closed_markets=topic.include_closed_markets,
            )
            _merge_discovered_markets(discovered_markets, related_markets)
            all_seed_tag_ids = sorted(set(all_seed_tag_ids) | set(related_subset))

    for condition_id in sorted(search_condition_ids):
        discovered_markets.setdefault(
            condition_id,
            DiscoveredMarket(
                condition_id=condition_id,
                question=None,
                tag_ids=[],
                tag_slugs=[],
                closed=None,
            ),
        )

    condition_ids = sorted(discovered_markets)
    logger.info(
        "Discovered %d condition IDs for topic '%s' from %d tags",
        len(condition_ids),
        topic_name,
        len(all_seed_tag_ids),
    )

    return DiscoveryResult(
        topic_name=topic_name,
        tag_ids=all_seed_tag_ids,
        condition_ids=condition_ids,
        markets=discovered_markets,
    )


async def _discover_markets_for_tags(
    *,
    gamma_client: GammaClient,
    tag_ids: list[int],
    include_closed_markets: bool,
) -> list[GammaMarket]:
    markets: list[GammaMarket] = []
    for tag_id in tag_ids:
        rows = await gamma_client.list_markets_by_tag(
            tag_id=tag_id,
            include_closed_markets=include_closed_markets,
        )
        markets.extend(rows)
    return markets


def _merge_discovered_markets(
    market_map: dict[str, DiscoveredMarket],
    new_markets: list[GammaMarket],
) -> None:
    for market in new_markets:
        condition_id = market.condition_id
        if not condition_id:
            continue
        row = market_map.get(condition_id)
        if row is None:
            market_map[condition_id] = DiscoveredMarket(
                condition_id=condition_id,
                question=market.question,
                tag_ids=sorted(set(market.tag_ids)),
                tag_slugs=sorted(set(market.tag_slugs)),
                closed=market.closed,
            )
            continue
        row.tag_ids = sorted(set(row.tag_ids) | set(market.tag_ids))
        row.tag_slugs = sorted(set(row.tag_slugs) | set(market.tag_slugs))
        if row.question is None and market.question:
            row.question = market.question
        if row.closed is None:
            row.closed = market.closed


def extract_tag_ids_from_search_row(row: dict[str, Any]) -> set[int]:
    tag_ids: set[int] = set()

    direct_tag_id = row.get("tag_id") or row.get("tagId")
    maybe_id = _as_int(direct_tag_id)
    if maybe_id is not None:
        tag_ids.add(maybe_id)

    row_type = str(row.get("type", "")).lower()
    if row_type == "tag":
        maybe_row_id = _as_int(row.get("id"))
        if maybe_row_id is not None:
            tag_ids.add(maybe_row_id)

    raw_tag_ids = row.get("tagIds") or row.get("tag_ids")
    if isinstance(raw_tag_ids, list):
        for value in raw_tag_ids:
            maybe = _as_int(value)
            if maybe is not None:
                tag_ids.add(maybe)

    raw_tags = row.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if not isinstance(tag, dict):
                continue
            maybe = _as_int(tag.get("id"))
            if maybe is not None:
                tag_ids.add(maybe)

    return tag_ids


def extract_condition_ids_from_search_row(row: dict[str, Any]) -> set[str]:
    condition_ids: set[str] = set()

    for key in ("conditionId", "condition_id", "market"):
        value = row.get(key)
        if isinstance(value, str) and value:
            condition_ids.add(value)

    raw_markets = row.get("markets")
    if isinstance(raw_markets, list):
        for market in raw_markets:
            if not isinstance(market, dict):
                continue
            value = market.get("conditionId") or market.get("condition_id")
            if isinstance(value, str) and value:
                condition_ids.add(value)

    return condition_ids


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
