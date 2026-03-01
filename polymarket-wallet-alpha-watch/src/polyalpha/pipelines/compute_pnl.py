from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from polyalpha.clients.data import DataClient, PositionRecord
from polyalpha.clients.gamma import GammaClient, GammaPublicProfile
from polyalpha.pipelines.collect_wallets import CandidateWallet
from polyalpha.pipelines.discover_markets import DiscoveredMarket

logger = logging.getLogger(__name__)

MARKET_CHUNK_SIZE = 60
REQUEST_BATCH_SIZE = 5
WALLET_BATCH_SIZE = 8


@dataclass(slots=True)
class MarketBreakdown:
    condition_id: str
    question: str | None
    invested: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass(slots=True)
class WalletPerformance:
    wallet_address: str
    name: str
    tags: list[str]
    wins: int
    losses: int
    win_rate: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_invested: float
    trade_count: int
    x_username: str | None = None
    verified_badge: bool = False
    top_markets: list[MarketBreakdown] = field(default_factory=list)


async def run_compute_pnl(
    *,
    data_client: DataClient,
    gamma_client: GammaClient,
    candidate_wallets: list[CandidateWallet],
    condition_ids: list[str],
    topic_name: str,
    markets: dict[str, DiscoveredMarket],
) -> list[WalletPerformance]:
    if not candidate_wallets:
        return []

    wallet_rows: list[WalletPerformance] = []
    condition_id_set = set(condition_ids)

    for i in range(0, len(candidate_wallets), WALLET_BATCH_SIZE):
        batch = candidate_wallets[i : i + WALLET_BATCH_SIZE]
        results = await asyncio.gather(
            *[
                _compute_wallet_performance(
                    data_client=data_client,
                    gamma_client=gamma_client,
                    candidate_wallet=wallet,
                    condition_ids=condition_ids,
                    condition_id_set=condition_id_set,
                    topic_name=topic_name,
                    market_lookup=markets,
                )
                for wallet in batch
            ]
        )
        wallet_rows.extend(results)

    wallet_rows.sort(
        key=lambda row: (
            -row.total_pnl,
            -row.realized_pnl,
            -row.total_invested,
            row.wallet_address,
        )
    )
    return wallet_rows


async def _compute_wallet_performance(
    *,
    data_client: DataClient,
    gamma_client: GammaClient,
    candidate_wallet: CandidateWallet,
    condition_ids: list[str],
    condition_id_set: set[str],
    topic_name: str,
    market_lookup: dict[str, DiscoveredMarket],
) -> WalletPerformance:
    wallet = candidate_wallet.wallet_address

    closed_positions = await _fetch_positions_chunked(
        data_client=data_client,
        wallet=wallet,
        condition_ids=condition_ids,
        fetch_closed=True,
    )
    open_positions = await _fetch_positions_chunked(
        data_client=data_client,
        wallet=wallet,
        condition_ids=condition_ids,
        fetch_closed=False,
    )

    # Defensive filter in case API ignores market filter for some rows.
    closed_positions = [pos for pos in closed_positions if (pos.resolved_condition_id() or "") in condition_id_set]
    open_positions = [pos for pos in open_positions if (pos.resolved_condition_id() or "") in condition_id_set]

    profile: GammaPublicProfile | None = None
    try:
        profile = await gamma_client.get_public_profile(wallet)
    except Exception as exc:  # pragma: no cover - external API variability
        logger.warning("Public profile fetch failed for %s: %s", wallet, exc)

    return aggregate_wallet_performance(
        wallet_address=wallet,
        closed_positions=closed_positions,
        open_positions=open_positions,
        topic_name=topic_name,
        trade_count=candidate_wallet.trade_count,
        profile=profile,
        fallback_name=candidate_wallet.observed_name,
        fallback_pseudonym=candidate_wallet.observed_pseudonym,
        market_lookup=market_lookup,
    )


async def _fetch_positions_chunked(
    *,
    data_client: DataClient,
    wallet: str,
    condition_ids: list[str],
    fetch_closed: bool,
) -> list[PositionRecord]:
    chunks = [condition_ids[i : i + MARKET_CHUNK_SIZE] for i in range(0, len(condition_ids), MARKET_CHUNK_SIZE)]
    all_positions: list[PositionRecord] = []

    for i in range(0, len(chunks), REQUEST_BATCH_SIZE):
        batch = chunks[i : i + REQUEST_BATCH_SIZE]
        if fetch_closed:
            responses = await asyncio.gather(
                *[
                    data_client.list_closed_positions(user=wallet, condition_ids=chunk)
                    for chunk in batch
                ]
            )
        else:
            responses = await asyncio.gather(
                *[data_client.list_positions(user=wallet, condition_ids=chunk) for chunk in batch]
            )
        for rows in responses:
            all_positions.extend(rows)

    return all_positions


def aggregate_wallet_performance(
    *,
    wallet_address: str,
    closed_positions: list[PositionRecord],
    open_positions: list[PositionRecord],
    topic_name: str,
    trade_count: int,
    profile: GammaPublicProfile | None,
    fallback_name: str | None,
    fallback_pseudonym: str | None,
    market_lookup: dict[str, DiscoveredMarket],
) -> WalletPerformance:
    wins = 0
    losses = 0
    realized = 0.0
    unrealized = 0.0
    invested = 0.0
    market_breakdown: dict[str, MarketBreakdown] = {}

    for position in closed_positions:
        condition_id = position.resolved_condition_id()
        if not condition_id:
            continue
        realized_value = _safe_float(position.realized_pnl)
        invested_value = _position_invested(position)

        realized += realized_value
        invested += invested_value

        if realized_value > 0:
            wins += 1
        elif realized_value < 0:
            losses += 1

        bucket = market_breakdown.setdefault(
            condition_id,
            MarketBreakdown(
                condition_id=condition_id,
                question=_resolve_question(condition_id, position.question, market_lookup),
            ),
        )
        bucket.invested += invested_value
        bucket.realized_pnl += realized_value

    for position in open_positions:
        condition_id = position.resolved_condition_id()
        if not condition_id:
            continue

        realized_component = _safe_float(position.realized_pnl)
        invested_value = _position_invested(position)
        unrealized_component = _position_unrealized(position, realized_component=realized_component)

        realized += realized_component
        invested += invested_value
        unrealized += unrealized_component

        bucket = market_breakdown.setdefault(
            condition_id,
            MarketBreakdown(
                condition_id=condition_id,
                question=_resolve_question(condition_id, position.question, market_lookup),
            ),
        )
        bucket.invested += invested_value
        bucket.realized_pnl += realized_component
        bucket.unrealized_pnl += unrealized_component

    denominator = wins + losses
    win_rate = (wins / denominator) if denominator else 0.0
    total_pnl = realized + unrealized

    display_name = _choose_display_name(
        profile=profile,
        fallback_name=fallback_name,
        fallback_pseudonym=fallback_pseudonym,
        wallet_address=wallet_address,
    )
    tags = _build_tags(profile=profile, topic_name=topic_name)
    top_markets = sorted(
        market_breakdown.values(),
        key=lambda row: (-abs(row.total_pnl), -row.total_pnl, row.condition_id),
    )[:5]

    return WalletPerformance(
        wallet_address=wallet_address,
        name=display_name,
        tags=tags,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_pnl=total_pnl,
        total_invested=invested,
        trade_count=trade_count,
        x_username=(profile.x_username if profile else None),
        verified_badge=(profile.verified_badge if profile else False),
        top_markets=top_markets,
    )


def _choose_display_name(
    *,
    profile: GammaPublicProfile | None,
    fallback_name: str | None,
    fallback_pseudonym: str | None,
    wallet_address: str,
) -> str:
    if profile:
        for value in (profile.name, profile.pseudonym, profile.x_username):
            if value:
                return value
    if fallback_name:
        return fallback_name
    if fallback_pseudonym:
        return fallback_pseudonym
    return wallet_address


def _build_tags(*, profile: GammaPublicProfile | None, topic_name: str) -> list[str]:
    tags = [topic_name]
    if profile:
        tags.extend(profile.tags)
        if profile.verified_badge:
            tags.append("verified")
    return sorted({tag for tag in tags if tag})


def _resolve_question(
    condition_id: str,
    position_question: str | None,
    market_lookup: dict[str, DiscoveredMarket],
) -> str | None:
    if position_question:
        return position_question
    market = market_lookup.get(condition_id)
    return market.question if market else None


def _position_unrealized(position: PositionRecord, *, realized_component: float) -> float:
    if position.cash_pnl is not None:
        return _safe_float(position.cash_pnl) - realized_component
    if position.current_value is not None and position.initial_value is not None:
        return _safe_float(position.current_value) - _safe_float(position.initial_value)
    return 0.0


def _position_invested(position: PositionRecord) -> float:
    if position.total_bought is not None:
        return _safe_float(position.total_bought)
    if position.initial_value is not None:
        return _safe_float(position.initial_value)
    return 0.0


def _safe_float(value: float | int | str | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
