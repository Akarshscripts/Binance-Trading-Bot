# Binance Trading Bot

A strategy-testing backend for algorithmic trading on **Binance** (crypto) and **Upstox** (Indian equities). Supports live signal streaming over WebSocket, historical backtesting with paper-trade simulation, and automated hyperparameter optimization via walk-forward validation.

---

## Features

- **Live signals** — streams real-time candles from the Binance WebSocket API; posts BUY/SELL signals with entry, stop-loss, and take-profit to a Discord webhook
- **Backtesting** — simulates a strategy on historical OHLCV data using a paper trader; reports 20+ performance metrics including win rate, profit factor, and trade duration
- **Quant research** — uses [Optuna](https://optuna.org/) to find optimal strategy parameters over a date range, with 3-phase walk-forward validation to prevent overfitting; persists trials to PostgreSQL so runs are resumable
- **Pluggable strategy interface** — all execution logic depends on the abstract `Strategy` base class; new strategies plug in by subclassing it without touching any backtest or prediction loop

---

## Architecture

```
CLI (Typer)
 ├── predict  ──►  commands/predict/   ──►  Strategy.process_candles()  ──►  Discord webhook
 ├── backtest ──►  commands/backtest/  ──►  Strategy.process_candles()  ──►  PaperTrader
 └── quant-research ─► commands/quant/ ──►  Optuna study (PostgreSQL)
                                               │
                        ┌──────────────────────┤
                        ▼                      ▼
               BinanceExchange          UpstoxExchange
               (HTTP + WebSocket)       (HTTP)
```

**Key layers:**

| Layer | Purpose |
|-------|---------|
| `strategies/` | `Strategy` ABC + `SupertrendStrategy` implementation |
| `indicators/` | Stateless indicator implementations (EMA, RSI, ADX, ATR, Supertrend, Fractals, Bollinger Bands, VWAP) |
| `brokers/` | `PaperTrader` — risk-based position sizing and P&L tracking |
| `binance_api/` | Binance REST + WebSocket data; auto-paginates large date ranges |
| `upstox_api/` | Upstox REST data; handles NSE trading-session timing |
| `messenger/` | Discord webhook sender |

---

## Setup

**Requirements:** Python ≥ 3.13, [`uv`](https://github.com/astral-sh/uv)

```bash
uv sync
```

Create a `.env` file in the project root:

```env
# Required only for quant-research (PostgreSQL connection string for Optuna)
OPTUNA_STORAGE_URL=postgresql://user:password@localhost:5432/optuna
```

---

## Usage

Run all commands via:

```bash
uv run python main.py <command> [args]
```

### `predict` — Live signal streaming

Connects to the Binance WebSocket stream and posts signals to Discord as each candle closes.

```bash
uv run python main.py predict BTCUSDT 15m --discord "https://discord.com/api/webhooks/..."
uv run python main.py predict ETHUSDT 1h  --discord "https://discord.com/api/webhooks/..."
```

### `backtest` — Historical simulation

```bash
uv run python main.py backtest BTCUSDT 15m \
  --start-time "01/01/2024 00:00:00" \
  --end-time   "04/01/2024 00:00:00"

# With custom strategy config and capital settings
uv run python main.py backtest TATA_STEEL 1h \
  --start-time "01/01/2024 00:00:00" \
  --end-time   "06/01/2024 00:00:00" \
  --capital 50000 \
  --risk-investment 0.01 \
  --config-file top_trials_TATA_STEEL_1h.json
```

Key options: `--capital` (default 10,000), `--brokerage` (default 0.1%), `--risk-investment` (fraction of capital risked per trade, default 2%), `--approval` (manually approve each trade).

### `quant-research` — Hyperparameter optimization

Runs an Optuna study to find the best strategy configuration, then cross-validates on earlier time windows to select the most robust parameters. Requires `OPTUNA_STORAGE_URL` in `.env`.

```bash
uv run python main.py quant-research BTCUSDT 15m \
  --start-time "01/01/2023 00:00:00" \
  --end-time   "01/01/2024 00:00:00" \
  --time-windows 5 \
  --n-trials 500 \
  --n-jobs -1
```

Saves the top 10 parameter sets to `top_trials_BTCUSDT_15m.json`, which can be passed to `backtest --config-file`.

---

## Supported Pairs

**Binance (crypto):**
`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `AVAXUSDT`, `LINKUSDT`, `DOTUSDT`, `MATICUSDT`, `LTCUSDT`, `TRXUSDT`, `UNIUSDT`, `ATOMUSDT`, `AAVEUSDT`, `BCHUSDT`, `FILUSDT`, `XLMUSDT`, `XMRUSDT`, `ALGOUSDT`, `XTZUSDT`

**Upstox (NSE equities):**
`TATA_STEEL`, `TCS`, `TATA_MOTORS`, `NESTLE_LTD`, `RELIANCE_POWER`, `VODAFONE_IDEA_LTD`, `IFCI_LTD`, `OBEROI_REALITY_LTD`

**Intervals:** `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d` (Binance); `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `1d` (Upstox)

---

## Strategy

`SupertrendStrategy` is the built-in strategy. It implements the `Strategy` interface and combines:

- **Supertrend** — primary trend direction; entry signal fires on a trend-direction flip
- **ADX** — filters out low-momentum markets; trades only when ADX > threshold (default 20)
- **Fractals** — sets stop-loss at the nearest fractal high/low; capped at 2% from entry
- **Dynamic R:R** — optionally scales the risk-reward ratio (1×–1.25×) based on EMA alignment, RSI momentum, and Bollinger Band position

### Adding a new strategy

1. Create a class that inherits from `strategies.Strategy`
2. Implement `process_candles(open, high, low, close, volume, round_off) -> PredictionOutput`
3. Swap the instantiation line in the relevant command file — the execution loop requires no other changes

---

## Performance Metrics

The backtest reports:

| Category | Metrics |
|----------|---------|
| Overview | Total trades, long/short split, win/loss count |
| P&L | Gross profit, gross loss, net P&L, brokerage breakdown |
| Per-trade | Avg profit, avg loss, max profit, max loss |
| Efficiency | Win rate, profit factor, avg/min/max trade duration (candles) |
| Risk | Grouped win % by risk-reward ratio |

Results are saved to `backtest_results.json`.

---

## Quant Research Pipeline

1. **Phase 1 — Optimization:** Runs `n_trials` Optuna trials on the most recent time window, maximising the Sharpe ratio. Trials are stored in PostgreSQL so the study can be resumed across runs.
2. **Phase 2 — Cross-validation:** Each completed trial is scored on all preceding time windows; the average score measures out-of-sample robustness.
3. **Phase 3 — Selection:** The top 10 trials by cross-validation score are saved to `top_trials_<SYMBOL>_<INTERVAL>.json` for use in production backtesting.
