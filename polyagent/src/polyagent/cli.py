from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich import print as rprint

from polyagent.config import Settings, load_config
from polyagent.data.loaders import load_market_data
from polyagent.data.make_demo_data import make_demo_data
from polyagent.env.replay_env import ReplayEnv

app = typer.Typer(help="PolyAgent research sandbox CLI.")
data_app = typer.Typer(help="Data utilities.")
app.add_typer(data_app, name="data")


@data_app.command("make-demo")
def data_make_demo(
    out: Path = typer.Option(Path("data/demo.parquet"), help="Output parquet/csv path."),
    n_steps: int = typer.Option(2000, help="Number of synthetic rows."),
    seed: int = typer.Option(7, help="Random seed."),
) -> None:
    path = make_demo_data(out_path=out, n_steps=n_steps, seed=seed)
    rprint(f"[green]Demo dataset created:[/green] {path}")


@app.command("train")
def train(
    data: Path = typer.Option(..., exists=True, help="Parquet/csv replay data path."),
    config: Path = typer.Option(Path("configs/mvp.yaml"), exists=True, help="Config file path."),
) -> None:
    cfg = load_config(config)
    df = load_market_data(data)
    env = ReplayEnv(df, cfg.env)
    obs, _ = env.reset(seed=cfg.seed)
    total_reward = 0.0
    for _ in range(min(128, cfg.env.episode_length)):
        action = int(env.action_space.sample())
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.logging.run_dir) / run_id
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"total_reward": total_reward, "obs_mean": float(obs.mean())}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ckpt_dir / "latest.pt").write_text("placeholder-checkpoint", encoding="utf-8")
    rprint(
        "[yellow]Step A skeleton training complete[/yellow] "
        f"(random policy). run_dir={run_dir} checkpoint={ckpt_dir / 'latest.pt'}"
    )


@app.command("eval")
def evaluate(
    data: Path = typer.Option(..., exists=True, help="Parquet/csv replay data path."),
    checkpoint: Path = typer.Option(..., exists=True, help="Checkpoint path."),
    config: Path = typer.Option(Path("configs/mvp.yaml"), exists=True, help="Config path."),
    use_search: bool = typer.Option(False, help="Enable decision-time search (Step D)."),
) -> None:
    _ = checkpoint
    _ = use_search
    cfg = load_config(config)
    df = load_market_data(data)
    env = ReplayEnv(df, cfg.env)
    _, _ = env.reset(seed=cfg.seed)
    total_reward = 0.0
    for _ in range(min(128, cfg.env.episode_length)):
        _, reward, done, _, _ = env.step(0)
        total_reward += reward
        if done:
            break
    rprint(f"[green]Eval complete[/green] reward={total_reward:.6f}")


@app.command("report")
def report(
    run_dir: Path = typer.Option(..., exists=True, help="Run directory containing metrics.json."),
    out: Path = typer.Option(Path("reports/latest_report.md"), help="Output markdown report."),
) -> None:
    settings = Settings()
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    lines = [
        f"# PolyAgent Report ({run_dir.name})",
        "",
        "## Summary",
        "Step A scaffold report. Detailed experiment management will be extended in later steps.",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"- openai_enabled: {bool(settings.openai_api_key)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]Report written:[/green] {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
