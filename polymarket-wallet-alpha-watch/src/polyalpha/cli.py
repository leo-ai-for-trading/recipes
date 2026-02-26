from __future__ import annotations

from pathlib import Path

import typer

from polyalpha.config import AppSettings, get_topic_or_raise, load_topics

app = typer.Typer(help="Polymarket wallet alpha watch CLI")


@app.command("topics")
def list_topics() -> None:
    """List configured topics."""
    settings = AppSettings()
    cfg = load_topics(Path(settings.topics_config_path))
    for topic_name in sorted(cfg.topics):
        typer.echo(topic_name)


@app.command("export")
def export(
    topic: str = typer.Option("google", help="Topic name from topics config."),
    out: Path = typer.Option(Path("reports/google.xlsx"), help="Output report path."),
    min_trades: int = typer.Option(5, help="Minimum trades to keep candidate wallet."),
    days: int | None = typer.Option(None, help="Optional lookback window in days."),
) -> None:
    """Run export pipeline (implemented in later steps)."""
    settings = AppSettings()
    cfg = load_topics(Path(settings.topics_config_path))
    _ = get_topic_or_raise(cfg, topic)
    typer.echo(
        "Scaffold ready. Pipeline implementation is pending Step B onward. "
        f"topic={topic} out={out} min_trades={min_trades} days={days}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
