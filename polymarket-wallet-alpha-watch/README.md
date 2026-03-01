# polymarket-wallet-alpha-watch

Analyze public Polymarket activity for a topic and export ranked **high-signal wallets** to Excel.

This tool does not label anyone as an insider or claim intent. It only analyzes public market/trade/position data.

## Features

- Topic config via YAML (`config/topics.yaml`)
- Market discovery from Gamma tags + public search fallback
- Candidate wallet discovery from Data API trades with pagination
- Per-wallet PnL aggregation (wins/losses, win rate, realized/unrealized, invested)
- Excel export with formatting:
  - `HighSignalWallets` (ranked summary)
  - optional `Evidence` (top market breakdown)
  - optional `Memos` (OpenAI-generated structured memo)
- Async clients with:
  - retry/backoff for 429/5xx and transport errors
  - request throttling and bounded concurrency

## Project Layout

```text
polymarket-wallet-alpha-watch/
  config/topics.yaml
  src/polyalpha/
    clients/
      base.py
      gamma.py
      data.py
      clob.py
    pipelines/
      discover_markets.py
      collect_wallets.py
      compute_pnl.py
      export_xlsx.py
    llm/openai_memo.py
    config.py
    cli.py
  tests/
  .env.example
  pyproject.toml
```

## Setup

```bash
cd polymarket-wallet-alpha-watch
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Optional OpenAI memo support:

```bash
pip install -e ".[llm]"
```

## Topics

Default topic config:

```yaml
topics:
  google:
    keywords: ["google", "GOOG", "GOOGL", "Google Search"]
    tag_slugs: ["google", "googl", "google-search"]
    tag_ids: []
    include_related_tags: true
    include_closed_markets: true
```

## CLI

List configured topics:

```bash
polyalpha topics
```

Export default report:

```bash
polyalpha export --topic google --out reports/google.xlsx --min-trades 5
```

Optional lookback filter for candidate trade discovery:

```bash
polyalpha export --topic google --days 365
```

Enable memo sheet:

```bash
polyalpha export --topic google --with-memos
```

## Output Columns

Sheet `HighSignalWallets`:

- `Rank`
- `Name`
- `Wallet Address`
- `Tags`
- `Wins`
- `Losses`
- `Win Rate`
- `Total P&L ($)`
- `Total Invested ($)`

## Demo Run

```bash
polyalpha export --topic google --out reports/google.xlsx --min-trades 5
```

## Notes and Caveats

- API fields may evolve; models are tolerant of unknown fields.
- If an endpoint returns partial/missing values, metrics gracefully default to zero where needed.
- `Unrealized PnL` uses:
  - `cashPnl - realizedPnl` when available
  - otherwise `currentValue - initialValue`
- Sorting is deterministic (`Total PnL`, then tie-breakers).
