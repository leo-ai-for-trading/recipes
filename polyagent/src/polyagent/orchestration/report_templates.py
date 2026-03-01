from __future__ import annotations

from datetime import datetime
from typing import Any


def render_report_markdown(
    *,
    run_id: str,
    metrics: dict[str, float],
    summary: str,
    next_runs: list[dict[str, Any]],
    llm_used: bool,
) -> str:
    lines = [
        f"# PolyAgent Report: {run_id}",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Summary",
        summary,
        "",
        "## Key Metrics",
    ]
    if metrics:
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            else:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- No metrics file found.")

    lines.extend(
        [
            "",
            "## Proposed Next Runs",
        ]
    )
    if next_runs:
        for item in next_runs:
            name = item.get("name", "unnamed-run")
            rationale = item.get("rationale", "")
            changes = item.get("changes", {})
            lines.append(f"- **{name}**: {rationale}")
            lines.append(f"  - changes: `{changes}`")
    else:
        lines.append("- No suggestions available.")

    lines.extend(
        [
            "",
            "## Governance Notes",
            "- LLM orchestration is optional and never in the high-frequency execution loop.",
            "- Strategy execution remains policy/belief/search only.",
            f"- llm_used: {llm_used}",
        ]
    )
    return "\n".join(lines) + "\n"
