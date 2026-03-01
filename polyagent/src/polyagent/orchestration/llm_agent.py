from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polyagent.config import Settings


SUGGESTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "next_runs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "changes": {"type": "object"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "changes", "rationale"],
            },
        },
    },
    "required": ["summary", "next_runs"],
}


def load_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    clean: dict[str, float] = {}
    for key, value in raw.items():
        try:
            clean[key] = float(value)
        except (TypeError, ValueError):
            continue
    return clean


def default_suggestions(metrics: dict[str, float]) -> dict[str, Any]:
    mean_reward = metrics.get("mean_batch_reward", 0.0)
    return {
        "summary": (
            "Baseline PPO-KL run completed. "
            "Use ablations to tune stability, sample efficiency, and execution risk."
        ),
        "next_runs": [
            {
                "name": "kl_tightened",
                "changes": {"rl.kl_target": 0.01, "rl.kl_coeff_init": 0.8},
                "rationale": "Reduce policy drift for more conservative updates.",
            },
            {
                "name": "search_on_eval",
                "changes": {"search.enabled": True, "search.n_rollouts": 32},
                "rationale": "Evaluate incremental value from decision-time improvement.",
            },
            {
                "name": "inventory_control",
                "changes": {"env.inventory_penalty": 0.001},
                "rationale": (
                    "If reward is unstable "
                    f"(mean_batch_reward={mean_reward:.4f}), tighten risk regularization."
                ),
            },
        ],
    }


def generate_orchestration_payload(
    *,
    settings: Settings,
    metrics: dict[str, float],
    run_notes: str,
) -> tuple[dict[str, Any], bool]:
    if not settings.openai_api_key:
        return default_suggestions(metrics), False

    try:
        from openai import OpenAI
    except ImportError:
        return default_suggestions(metrics), False

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an experiment orchestrator for an RL trading research sandbox. "
                        "Suggest controlled experiments only. "
                        "Do not issue trading actions."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"metrics": metrics, "notes": run_notes}),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "experiment_suggestions",
                    "schema": SUGGESTIONS_SCHEMA,
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
        return payload, True
    except Exception:
        return default_suggestions(metrics), False


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
