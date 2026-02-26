# polymarket-wallet-alpha-watch

Track high-signal wallets trading a topic cluster on Polymarket, compute wallet-level PnL metrics, and export ranked results to Excel.

## Scope (MVP)
- Discover markets by topic (keywords + tags).
- Discover candidate wallets from topic market trades.
- Compute wallet metrics from closed/open positions.
- Export ranked `.xlsx` report.
- Optional OpenAI investigation memo sheet (off by default).

## Project Layout
```text
polymarket-wallet-alpha-watch/
  config/topics.yaml
  src/polyalpha/
    clients/{gamma.py,data.py,clob.py}
    pipelines/{discover_markets.py,collect_wallets.py,compute_pnl.py,export_xlsx.py}
    llm/openai_memo.py
    config.py
    cli.py
  tests/
```

## Quickstart
```bash
cd polymarket-wallet-alpha-watch
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Commands
```bash
polyalpha topics
polyalpha export --topic google --out reports/google.xlsx --min-trades 5
```

## Configuration
- Topic definitions live in `config/topics.yaml`.
- Runtime env vars are documented in `.env.example`.

## Notes
- This tool is for analysis/research workflows.
- API responses/fields can evolve; clients should be version-tolerant.
