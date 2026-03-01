# polyagent

Research-grade **agentic trading** sandbox for prediction markets with strict separation between:

- **Execution core (fast loop):** RL policy + belief model + optional decision-time search
- **Orchestration layer (slow loop):** LLM for monitoring, summaries, and experiment suggestions

LLM is never used for high-frequency action selection.

## Ethics and Scope

- This is a research simulator for public market data replay.
- No real-money execution in this repository.
- Reports are analytical and do not claim illicit intent.

## Design Anchor

- Paper: `docs/papers/2511.07312v1.pdf`
- Mapping notes: `design_notes.md`

If the PDF is missing, add it at that path.

## Architecture

`src/polyagent/`

- `env/` replay simulator, fills, fee model
- `belief/` particle belief state + likelihood net
- `models/` policy/value networks
- `rl/` PPO with explicit KL regularization and advantage filtering support
- `search/` decision-time policy improvement under KL cap
- `orchestration/` optional OpenAI-powered reporting/suggestions
- `data/` loaders + demo data generation
- `cli.py` commands

## Requirements

- Python 3.11+
- CPU is enough for MVP

## Install

```bash
cd polyagent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Config

Main config: `configs/mvp.yaml`

Includes:
- `env` tick, levels, fees, episode params
- `rl` PPO/KL hyperparameters
- `belief` particle filter settings
- `search` rollout and KL-cap parameters
- `logging` tensorboard/checkpoint dirs
- `advantage_filter` quantile/min threshold

## CLI Commands

Generate demo replay data:

```bash
polyagent data make-demo --out data/demo.parquet
```

Train (PPO+KL):

```bash
polyagent train --data data/demo.parquet --config configs/mvp.yaml
```

Evaluate:

```bash
polyagent eval --data data/demo.parquet --checkpoint checkpoints/latest.pt --use-search true
```

Generate report (non-LLM always, LLM optional if `OPENAI_API_KEY` set):

```bash
polyagent report --run-dir runs/<run_id>
```

## Notes on Execution Core

- Observation: top-of-book levels, trade summary, inventory/time features.
- Action space (discrete): hold, post bid/ask variants, cancel.
- Reward: step PnL - inventory penalty - impact penalty - fees.
- PPO includes explicit KL penalty with adaptive coefficient.
- Decision-time search computes:
  - `pi_search ∝ pi_base * exp(Q / alpha_kl)`
  - then KL-caps via blending.

## Tests

```bash
ruff format .
ruff check .
pytest
```

Core tests:
- reward/fill behavior
- advantage filtering
- decision-time search KL cap
- replay env step contract
