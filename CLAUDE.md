# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management (Python >=3.13 required).

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py --help

# Key CLI commands
uv run python main.py predict <PAIR> <INTERVAL> --discord <WEBHOOK_URL>
uv run python main.py backtest <PAIR> <INTERVAL> --start-time "MM/dd/yyyy HH:mm:ss" --end-time "MM/dd/yyyy HH:mm:ss"
uv run python main.py quant-research <PAIR> <INTERVAL> --start-time "..." --end-time "..."

# Format code
uv run black .
```

## Environment Variables

Create a `.env` file with:
- `OPTUNA_STORAGE_URL` — PostgreSQL connection string for Optuna study persistence (required for `quant-research`)

## Architecture

The app is a rule-based algorithmic trading bot with three CLI commands wired through `typer`:

**Entry flow:** `main.py` → `cli/app.py` (Typer root) → sub-commands in `cli/` → `commands/` for business logic.

### CLI Layer (`cli/`)
Each command has its own module (`backtest.py`, `predict.py`, `research.py`) that parses args and delegates to `commands/`. `cli/models.py` defines `PairRegistry`, which maps exchange symbol enums to exchange classes — new exchanges must be registered here.

### Commands Layer (`commands/`)
Three command implementations:
- `commands/backtest/` — runs historical simulation using `PaperTrader`
- `commands/predict/` — live real-time prediction loop posting signals to Discord
- `commands/quant/` — Optuna-based hyperparameter optimization with walk-forward validation

The quant research flow is a 3-phase process: (1) optimize strategy params on the **most recent** time window using Optuna + PostgreSQL storage, (2) cross-validate all completed trials on **preceding** windows, (3) dump the top 10 trials by average cross-validation score to `top_trials_<SYMBOL>_<INTERVAL>.json`.

### Strategy Layer (`strategies/`)
`SupertrendStrategy` is the only strategy. It combines Supertrend + Fractals for entries/exits, ADX for trend strength filter, and optionally uses EMA alignment + RSI + Bollinger Bands for dynamic risk-reward scaling. All configuration lives in `SupertrendStrategyConfig` (Pydantic model); configs can be loaded from / saved to JSON files.

### Brokers Layer (`brokers/`)
`PaperTrader` simulates trades without real execution. Position sizing is risk-based: it calculates shares from `risk_per_trade / risk_per_share`, capped by available capital. The `stats()` method returns `PaperTradeStats` with full P&L breakdown, grouped by risk-reward ratio.

### Indicators Layer (`indicators/`)
All indicators inherit from `Indicator` (abstract base in `abstract.py`) and must declare a `NAME` from `AvailableIndicators`. Each indicator is stateless — `get_value()` accepts price lists and returns computed values. Available: EMA, RSI, ADX, ATR, VWAP, Fractals, SuperTrend, BollingerBands.

### Exchange Adapters (`binance_api/`, `upstox_api/`)
Each adapter exposes an exchange class (e.g., `BinanceExchange`) and symbol/interval enums. `BinanceExchange.get_symbol_info()` paginates automatically across large date ranges, rate-limiting itself after 3000 requests.

### Messenger (`messenger/`)
Sends trade signals to Discord webhooks. `exception_handler.py` wraps calls with retry/error reporting.

## Conventions

- Supported trading pairs are defined as `Enum` members in `binance_api/models.py` and `upstox_api/models.py`. To add a new symbol, add it to the appropriate enum.
- Strategy configs for backtesting can be passed via `--config-file` pointing to a JSON file matching `SupertrendStrategyConfig` fields.
- Logs are written to `logs/app.log` and stdout.
