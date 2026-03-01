from __future__ import annotations

from datetime import datetime


def base_report_template(run_id: str, summary: str, metrics: dict[str, float]) -> str:
    lines = [
        f"# PolyAgent Report: {run_id}",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Summary",
        summary,
        "",
        "## Metrics",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"- {key}: {value:.6f}")
    return "\n".join(lines) + "\n"
