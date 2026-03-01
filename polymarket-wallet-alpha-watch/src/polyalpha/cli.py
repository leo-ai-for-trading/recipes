from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from polyalpha.clients.data import DataClient
from polyalpha.clients.gamma import GammaClient
from polyalpha.config import AppSettings, TopicConfig, get_topic_or_raise, load_topics
from polyalpha.llm import generate_memos, memos_to_rows
from polyalpha.pipelines.collect_wallets import run_collect_wallets
from polyalpha.pipelines.compute_pnl import run_compute_pnl
from polyalpha.pipelines.discover_markets import run_discover_markets
from polyalpha.pipelines.export_xlsx import build_evidence_rows, run_export_xlsx

app = typer.Typer(help="Polymarket wallet alpha watch CLI")
logger = logging.getLogger(__name__)


@app.command("topics")
def list_topics() -> None:
    """List configured topics."""
    try:
        settings = AppSettings()
        cfg = load_topics(Path(settings.topics_config_path))
        for topic_name in sorted(cfg.topics):
            typer.echo(topic_name)
    except Exception as exc:
        typer.echo(f"Failed to load topics: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("export")
def export(
    topic: str = typer.Option("google", help="Topic name from topics config."),
    out: Path = typer.Option(Path("reports/google.xlsx"), help="Output report path."),
    min_trades: int = typer.Option(5, help="Minimum trades to keep candidate wallet."),
    days: int | None = typer.Option(None, help="Optional lookback window in days."),
    evidence: bool = typer.Option(True, "--evidence/--no-evidence", help="Include Evidence sheet."),
    with_memos: bool = typer.Option(
        False,
        "--with-memos/--no-memos",
        help="Generate optional AI memo sheet (requires OPENAI_API_KEY).",
    ),
) -> None:
    """Export ranked high-signal wallets for a topic."""
    _configure_logging()
    try:
        settings = AppSettings()
        cfg = load_topics(Path(settings.topics_config_path))
        topic_cfg = get_topic_or_raise(cfg, topic)

        result_path = asyncio.run(
            _run_export(
                settings=settings,
                topic_name=topic,
                topic_cfg=topic_cfg,
                out_path=out,
                min_trades=min_trades,
                days=days,
                include_evidence=evidence,
                include_memos=with_memos,
            )
        )
        typer.echo(f"Export completed: {result_path}")
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()


async def _run_export(
    *,
    settings: AppSettings,
    topic_name: str,
    topic_cfg: TopicConfig,
    out_path: Path,
    min_trades: int,
    days: int | None,
    include_evidence: bool,
    include_memos: bool,
) -> Path:
    gamma = GammaClient(
        base_url=settings.gamma_base_url,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.request_max_retries,
        backoff_seconds=settings.request_backoff_seconds,
        max_concurrency=settings.max_concurrency,
        requests_per_second=settings.requests_per_second,
    )
    data = DataClient(
        base_url=settings.data_base_url,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.request_max_retries,
        backoff_seconds=settings.request_backoff_seconds,
        max_concurrency=settings.max_concurrency,
        requests_per_second=settings.requests_per_second,
    )

    async with gamma, data:
        discovery = await run_discover_markets(
            topic_name=topic_name,
            topic=topic_cfg,
            gamma_client=gamma,
        )
        if not discovery.condition_ids:
            raise RuntimeError(f"No markets found for topic '{topic_name}'.")

        candidates = await run_collect_wallets(
            data_client=data,
            condition_ids=discovery.condition_ids,
            min_trades=min_trades,
            days=days,
        )
        if not candidates:
            raise RuntimeError("No candidate wallets found with the provided filters.")

        wallet_rows = await run_compute_pnl(
            data_client=data,
            gamma_client=gamma,
            candidate_wallets=candidates,
            condition_ids=discovery.condition_ids,
            topic_name=topic_name,
            markets=discovery.markets,
        )

    evidence_rows = build_evidence_rows(wallet_rows) if include_evidence else None
    memo_rows = None
    if include_memos:
        memo_settings = settings.model_copy(update={"openai_memo_enabled": True})
        memos = generate_memos(wallet_rows=wallet_rows, settings=memo_settings)
        memo_rows = memos_to_rows(memos)

    run_export_xlsx(
        wallet_rows=wallet_rows,
        out_path=out_path,
        evidence_rows=evidence_rows,
        memo_rows=memo_rows,
    )
    return out_path


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
