# polyagent

Research-grade sandbox for agentic trading architecture on prediction markets.

## Scope
- Offline replay environment on historical/synthetic order book + trade data.
- RL execution core (policy/value + PPO-KL + optional decision-time search).
- Belief modeling for imperfect information.
- Optional LLM orchestration for reports/experiment suggestions.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
polyagent data make-demo --out data/demo.parquet
polyagent train --data data/demo.parquet --config configs/mvp.yaml
polyagent eval --data data/demo.parquet --checkpoint checkpoints/latest.pt --use-search true
polyagent report --run-dir runs/latest
```

## Paper Anchor
Design references are documented in `design_notes.md`, aligned with:
- `docs/papers/2511.07312v1.pdf`
