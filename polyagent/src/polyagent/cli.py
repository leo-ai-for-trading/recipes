from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import typer
from rich import print as rprint

from polyagent.belief.likelihood_net import LikelihoodNet
from polyagent.belief.belief_state import BeliefState
from polyagent.belief.particle_filter import ParticleBeliefModel
from polyagent.config import Settings, apply_dot_overrides, load_config, save_config
from polyagent.data.loaders import load_market_data
from polyagent.data.make_demo_data import make_demo_data
from polyagent.env.replay_env import ReplayEnv
from polyagent.models.policy_net import PolicyNet
from polyagent.models.value_net import ValueNet
from polyagent.orchestration.llm_agent import (
    generate_orchestration_payload,
    load_metrics,
    save_json_report,
)
from polyagent.orchestration.report_templates import render_report_markdown
from polyagent.rl.buffers import RolloutBuffer
from polyagent.rl.ppo_kl import PPOKLTrainer
from polyagent.rl.utils import advantage_filter_mask, compute_gae, normalize, set_seed
from polyagent.search.decision_time_search import DecisionTimeSearcher

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


@app.command("improve-from-report")
def improve_from_report(
    report_json: Path = typer.Option(..., exists=True, help="Report JSON with next_runs."),
    base_config: Path = typer.Option(
        Path("configs/mvp.yaml"), exists=True, help="Base config yaml to modify."
    ),
    out: Path = typer.Option(
        Path("configs/mvp_improved.yaml"),
        help="Output config path with applied report suggestions.",
    ),
) -> None:
    cfg = load_config(base_config)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    next_runs = payload.get("next_runs", [])
    if not isinstance(next_runs, list):
        raise typer.Exit(code=1)

    merged_overrides: dict[str, object] = {}
    for item in next_runs:
        if not isinstance(item, dict):
            continue
        changes = item.get("changes", {})
        if isinstance(changes, dict):
            merged_overrides.update(changes)

    if not merged_overrides:
        rprint("[yellow]No overrides found in report JSON. Nothing changed.[/yellow]")
        return

    improved = apply_dot_overrides(cfg, merged_overrides)
    saved = save_config(improved, out)
    rprint(f"[green]Improved config written:[/green] {saved}")
    rprint(f"[cyan]Applied overrides:[/cyan] {json.dumps(merged_overrides, indent=2)}")


@app.command("train")
def train(
    data: Path = typer.Option(..., exists=True, help="Parquet/csv replay data path."),
    config: Path = typer.Option(Path("configs/mvp.yaml"), exists=True, help="Config file path."),
) -> None:
    cfg = load_config(config)
    settings = Settings()
    set_seed(cfg.seed)

    df = load_market_data(data)
    env = ReplayEnv(df, cfg.env)
    raw_obs_dim = int(env.observation_space.shape[0])
    belief_feature_dim = cfg.belief.latent_dim * 2
    obs_dim = raw_obs_dim + belief_feature_dim
    action_dim = int(env.action_space.n)

    device = torch.device("cpu")
    policy = PolicyNet(obs_dim=obs_dim, hidden_dim=cfg.model.hidden_dim, action_dim=action_dim).to(device)
    value_net = ValueNet(obs_dim=obs_dim, hidden_dim=cfg.model.hidden_dim).to(device)
    trainer = PPOKLTrainer(policy=policy, value_net=value_net, cfg=cfg.rl, device=device)
    buffer = RolloutBuffer()
    likelihood_net = LikelihoodNet(
        obs_dim=raw_obs_dim,
        latent_dim=cfg.belief.latent_dim,
        hidden_dims=cfg.belief.likelihood_hidden_dims,
    )
    belief_model = ParticleBeliefModel(
        n_particles=cfg.belief.n_particles,
        latent_dim=cfg.belief.latent_dim,
        transition_noise=cfg.belief.transition_noise,
        likelihood_net=likelihood_net,
        seed=cfg.seed,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.logging.run_dir) / run_id
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = _maybe_tensorboard_writer(
        enabled=(cfg.logging.tensorboard and settings.tensorboard_enabled),
        log_dir=run_dir / "tb",
    )

    raw_obs, _ = env.reset(seed=cfg.seed)
    belief = belief_model.init_belief()
    obs = _augment_obs(raw_obs=raw_obs, belief_model=belief_model, belief=belief)
    train_rewards: list[float] = []
    for update in range(cfg.rl.train_steps):
        buffer.clear()
        batch_reward = 0.0
        for _ in range(cfg.rl.batch_size):
            action, log_prob, value, logits = trainer.action_and_value(obs)
            next_raw_obs, reward, done, _, _ = env.step(action)
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
                logits=logits,
            )
            belief = belief_model.update(belief, next_raw_obs)
            obs = _augment_obs(raw_obs=next_raw_obs, belief_model=belief_model, belief=belief)
            batch_reward += reward
            if done:
                raw_obs, _ = env.reset(seed=cfg.seed)
                belief = belief_model.init_belief()
                obs = _augment_obs(raw_obs=raw_obs, belief_model=belief_model, belief=belief)

        arrays = buffer.as_arrays()
        last_value = trainer.value(obs)
        advantages, returns = compute_gae(
            rewards=arrays["rewards"],
            values=arrays["values"],
            dones=arrays["dones"],
            last_value=last_value,
            gamma=cfg.rl.gamma,
            lam=cfg.rl.lam,
        )
        arrays["advantages"] = normalize(advantages)
        arrays["returns"] = returns
        mask = advantage_filter_mask(
            arrays["advantages"],
            enabled=cfg.advantage_filter.enabled,
            quantile=cfg.advantage_filter.quantile,
            min_abs_adv=cfg.advantage_filter.min_abs_adv,
        )
        filtered = {key: value[mask] for key, value in arrays.items()}
        kept_ratio = float(mask.mean())
        stats = trainer.update(filtered)

        train_rewards.append(batch_reward)
        if writer is not None:
            writer.add_scalar("train/reward_batch", batch_reward, update)
            writer.add_scalar("train/loss_total", stats.total_loss, update)
            writer.add_scalar("train/loss_policy", stats.policy_loss, update)
            writer.add_scalar("train/loss_value", stats.value_loss, update)
            writer.add_scalar("train/entropy", stats.entropy, update)
            writer.add_scalar("train/kl", stats.kl, update)
            writer.add_scalar("train/kl_coeff", stats.kl_coeff, update)
            writer.add_scalar("train/adv_kept_ratio", kept_ratio, update)
            writer.add_scalar("train/early_stopped", int(stats.early_stopped), update)

        rprint(
            f"[cyan]update {update + 1}/{cfg.rl.train_steps}[/cyan] "
            f"reward_batch={batch_reward:.4f} kl={stats.kl:.5f} "
            f"adv_kept={kept_ratio:.2f} early_stop={stats.early_stopped}"
        )

    if writer is not None:
        writer.flush()
        writer.close()

    checkpoint = {
        "policy_state": policy.state_dict(),
        "value_state": value_net.state_dict(),
        "likelihood_state": likelihood_net.state_dict(),
        "obs_dim": obs_dim,
        "raw_obs_dim": raw_obs_dim,
        "action_dim": action_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "belief_latent_dim": cfg.belief.latent_dim,
        "belief_hidden_dims": cfg.belief.likelihood_hidden_dims,
        "config_path": str(config),
    }
    latest_ckpt = ckpt_dir / "latest.pt"
    torch.save(checkpoint, latest_ckpt)
    torch.save(checkpoint, ckpt_dir / f"{run_id}.pt")

    metrics = {
        "mean_batch_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
        "last_batch_reward": float(train_rewards[-1]) if train_rewards else 0.0,
        "n_updates": cfg.rl.train_steps,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.txt").write_text(str(latest_ckpt), encoding="utf-8")
    rprint(f"[green]Training complete[/green] run_dir={run_dir} checkpoint={latest_ckpt}")


@app.command("eval")
def evaluate(
    data: Path = typer.Option(..., exists=True, help="Parquet/csv replay data path."),
    checkpoint: Path = typer.Option(..., exists=True, help="Checkpoint path."),
    config: Path = typer.Option(Path("configs/mvp.yaml"), exists=True, help="Config path."),
    use_search: bool = typer.Option(False, help="Enable decision-time search."),
) -> None:
    _ = use_search
    cfg = load_config(config)
    set_seed(cfg.seed)

    df = load_market_data(data)
    env = ReplayEnv(df, cfg.env)
    ckpt = torch.load(checkpoint, map_location="cpu")
    policy = PolicyNet(
        obs_dim=int(ckpt["obs_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        action_dim=int(ckpt["action_dim"]),
    )
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    value_net = ValueNet(
        obs_dim=int(ckpt["obs_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    )
    value_net.load_state_dict(ckpt["value_state"])
    value_net.eval()
    likelihood_net = LikelihoodNet(
        obs_dim=int(ckpt["raw_obs_dim"]),
        latent_dim=int(ckpt["belief_latent_dim"]),
        hidden_dims=list(ckpt["belief_hidden_dims"]),
    )
    if "likelihood_state" in ckpt:
        likelihood_net.load_state_dict(ckpt["likelihood_state"])
    belief_model = ParticleBeliefModel(
        n_particles=cfg.belief.n_particles,
        latent_dim=int(ckpt["belief_latent_dim"]),
        transition_noise=cfg.belief.transition_noise,
        likelihood_net=likelihood_net,
        seed=cfg.seed,
    )
    search_enabled = use_search or cfg.search.enabled
    searcher = (
        DecisionTimeSearcher(
            cfg=cfg.search,
            env_cfg=cfg.env,
            belief_model=belief_model,
            value_net=value_net,
            device=torch.device("cpu"),
            seed=cfg.seed,
        )
        if search_enabled
        else None
    )

    raw_obs, _ = env.reset(seed=cfg.seed)
    belief = belief_model.init_belief()
    obs = _augment_obs(raw_obs=raw_obs, belief_model=belief_model, belief=belief)
    total_reward = 0.0
    steps = 0
    kls: list[float] = []
    done = False
    while not done and steps < cfg.env.episode_length:
        if searcher is not None:
            search_out = searcher.run(policy=policy, raw_obs=raw_obs, belief=belief)
            action = int(np.argmax(search_out.pi_search))
            kls.append(search_out.kl)
        else:
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
        next_raw_obs, reward, done, _, _ = env.step(action)
        belief = belief_model.update(belief, next_raw_obs)
        obs = _augment_obs(raw_obs=next_raw_obs, belief_model=belief_model, belief=belief)
        raw_obs = next_raw_obs
        total_reward += reward
        steps += 1

    avg_kl = float(np.mean(kls)) if kls else 0.0
    rprint(
        f"[green]Eval complete[/green] reward={total_reward:.6f} "
        f"steps={steps} search={search_enabled} avg_search_kl={avg_kl:.6f}"
    )


@app.command("report")
def report(
    run_dir: Path = typer.Option(..., exists=True, help="Run directory containing metrics.json."),
    out: Path | None = typer.Option(None, help="Output markdown report path."),
) -> None:
    settings = Settings()
    metrics = load_metrics(run_dir)
    payload, llm_used = generate_orchestration_payload(
        settings=settings,
        metrics=metrics,
        run_notes="Local report generation from run metrics.",
    )
    summary = str(payload.get("summary", "No summary available."))
    next_runs = payload.get("next_runs", [])
    if not isinstance(next_runs, list):
        next_runs = []

    report_path = out or Path("reports") / f"{run_dir.name}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = render_report_markdown(
        run_id=run_dir.name,
        metrics=metrics,
        summary=summary,
        next_runs=next_runs,
        llm_used=llm_used,
    )
    report_path.write_text(report_md, encoding="utf-8")
    save_json_report(report_path.with_suffix(".json"), payload)
    rprint(f"[green]Report written:[/green] {report_path}")


def _maybe_tensorboard_writer(enabled: bool, log_dir: Path):  # type: ignore[no-untyped-def]
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _augment_obs(
    *,
    raw_obs: np.ndarray,
    belief_model: ParticleBeliefModel,
    belief: BeliefState,
) -> np.ndarray:
    belief_features = belief_model.features(belief)
    return np.concatenate([raw_obs.astype(np.float32), belief_features], axis=0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
