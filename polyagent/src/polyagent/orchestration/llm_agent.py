from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polyagent.config import Settings


def summarize_run_with_llm(
    *,
    settings: Settings,
    metrics: dict[str, float],
    run_notes: str,
) -> dict[str, Any] | None:
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=settings.openai_api_key)
    schema = {
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
    resp = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a research orchestrator. Propose experiments only. "
                    "You are not in the execution loop."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"metrics": metrics, "notes": run_notes}),
            },
        ],
        text={"format": {"type": "json_schema", "name": "next_runs", "schema": schema, "strict": True}},
    )
    return json.loads(resp.output_text)


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
