from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from polyalpha.clients.data import DataClient, TradeRecord

logger = logging.getLogger(__name__)

MARKET_CHUNK_SIZE = 40
REQUEST_BATCH_SIZE = 5


@dataclass(slots=True)
class CandidateWallet:
    wallet_address: str
    trade_count: int
    observed_name: str | None = None
    observed_pseudonym: str | None = None


async def run_collect_wallets(
    *,
    data_client: DataClient,
    condition_ids: list[str],
    min_trades: int = 5,
    days: int | None = None,
) -> list[CandidateWallet]:
    if not condition_ids:
        return []

    cutoff: datetime | None = None
    if days is not None:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)

    trade_counts: dict[str, int] = {}
    names: dict[str, str] = {}
    pseudonyms: dict[str, str] = {}

    chunks = [condition_ids[i : i + MARKET_CHUNK_SIZE] for i in range(0, len(condition_ids), MARKET_CHUNK_SIZE)]
    logger.info("Collecting candidate wallets from %d market chunks", len(chunks))

    for i in range(0, len(chunks), REQUEST_BATCH_SIZE):
        batch = chunks[i : i + REQUEST_BATCH_SIZE]
        responses = await asyncio.gather(
            *[data_client.list_trades(condition_ids=chunk) for chunk in batch]
        )
        for trades in responses:
            _consume_trade_batch(
                trades=trades,
                cutoff=cutoff,
                trade_counts=trade_counts,
                names=names,
                pseudonyms=pseudonyms,
            )

    rows = [
        CandidateWallet(
            wallet_address=wallet,
            trade_count=count,
            observed_name=names.get(wallet),
            observed_pseudonym=pseudonyms.get(wallet),
        )
        for wallet, count in trade_counts.items()
        if count >= min_trades
    ]
    rows.sort(key=lambda row: (-row.trade_count, row.wallet_address))

    logger.info("Found %d candidate wallets (min_trades=%d)", len(rows), min_trades)
    return rows


def _consume_trade_batch(
    *,
    trades: list[TradeRecord],
    cutoff: datetime | None,
    trade_counts: dict[str, int],
    names: dict[str, str],
    pseudonyms: dict[str, str],
) -> None:
    for trade in trades:
        if cutoff is not None and trade.timestamp is not None and trade.timestamp < cutoff:
            continue

        wallet = _normalize_wallet(trade.proxy_wallet)
        if wallet is None:
            continue

        trade_counts[wallet] = trade_counts.get(wallet, 0) + 1

        extras = trade.model_extra or {}
        maybe_name = _first_non_empty(
            [
                extras.get("name"),
                extras.get("displayName"),
                extras.get("display_name"),
            ]
        )
        maybe_pseudonym = _first_non_empty(
            [
                extras.get("pseudonym"),
                extras.get("username"),
                extras.get("xUsername"),
                extras.get("x_username"),
            ]
        )
        if maybe_name and wallet not in names:
            names[wallet] = maybe_name
        if maybe_pseudonym and wallet not in pseudonyms:
            pseudonyms[wallet] = maybe_pseudonym


def _normalize_wallet(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    wallet = value.strip().lower()
    if not wallet:
        return None
    return wallet


def _first_non_empty(values: list[object]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
