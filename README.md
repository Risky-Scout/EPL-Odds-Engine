# EPL Joint PMF Research Engine

A production research engine for estimating the **joint probability mass function of final scores** in the English Premier League:

\[
P(G_h = x, G_a = y)
\]

This repo is built around one object: the full home-away score grid. From that single grid, the engine derives:

- 1X2
- totals
- BTTS
- correct score
- Asian-handicap-style margin summaries
- model-vs-market edge
- pre-game and live in-play distributions

## Mission

Build a model that aims to:

- estimate true probabilities and distributions better than the market
- stay well-calibrated over time
- identify positive-EV bets responsibly
- publish transparent, reproducible probability outputs to GitHub and WizardOfOdds
- support both **pre-game** and **live in-play** forecasting

This repo does **not** assume success. Market-beating claims must be earned through walk-forward evidence, calibration diagnostics, closing-line tracking, and live or paper-trading logs.

## What is implemented

### Historical research engine
- Penaltyblog historical ingestion via `pb.scrapers.FootballData`
- walk-forward splits with train / validation / test windows
- time-decayed model fitting via `pb.models.dixon_coles_weights`
- model ensemble across:
  - Dixon-Coles
  - Bivariate Poisson
  - Zero-Inflated Poisson
  - Negative Binomial
  - Weibull-Copula
- Pi ratings prior via `pb.ratings.PiRatingSystem`
- validation-based ensemble weighting
- score-grid temperature calibration
- market comparison with no-vig implied probabilities
- CLV / EV / Kelly-ready outputs
- publication-grade figures and HTML / Markdown reports

### Live engine
- BALLDONTLIE EPL adapters for:
  - matches
  - match events
  - match lineups
  - team match stats
  - odds
- cache-aware, rate-limit-aware client
- live state updater that transforms a pre-game PMF into an in-play PMF
- WizardOfOdds JSON export

## Why this split exists

`penaltyblog` is the modeling layer. Its docs show football-data.co.uk scrapers, Pi ratings, multiple goal models, implied odds tools, and backtesting-oriented workflows. `BALLDONTLIE` is useful for live EPL operations, but the official EPL odds endpoint is limited to moneyline home / draw / away odds, so it should not be the only historical source when your target is the full scoreline distribution. See the sources section at the bottom of this README for the current documentation links.

## Repository layout

```text
epl_joint_pmf_github_repo/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── sample_outputs/
├── src/epl_pmf_backtest/
│   ├── providers/
│   ├── backtest.py
│   ├── calibration.py
│   ├── cli.py
│   ├── config.py
│   ├── features.py
│   ├── live.py
│   ├── markets.py
│   ├── modeling.py
│   ├── reporting.py
│   ├── storage.py
│   ├── types.py
│   └── wizardofodds.py
├── tests/
├── pyproject.toml
├── Makefile
└── .gitignore
```

## Quick start

```bash
git clone <your-repo-url>
cd epl_joint_pmf_github_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
cp configs/epl_walkforward.example.yaml configs/epl_walkforward.yaml
```

Historical sync:

```bash
make sync CONFIG=configs/epl_walkforward.yaml
```

Walk-forward backtest:

```bash
make backtest CONFIG=configs/epl_walkforward.yaml
```

Report generation:

```bash
make report CONFIG=configs/epl_walkforward.yaml
```

Live snapshot:

```bash
make live CONFIG=configs/live.example.yaml
```

## Main artifacts

A real backtest run writes reproducible artifacts under the configured storage root:

- `raw/`
- `curated/`
- `features/`
- `backtests/`
- `reports/`
- `figures/`
- `live/`
- `exports/`
- `cache/http/`

## Important implementation note

The backtest engine is coded against the current Penaltyblog and BALLDONTLIE docs, but it was not executed end-to-end inside this chat environment because the environment is not connected to your real API key or a local `penaltyblog` runtime. The repo is designed to run locally or on your server with real data.

## Sources

- Penaltyblog docs: models, Pi ratings, scrapers, implied odds, backtesting
- Penaltyblog GitHub repo and release history
- BALLDONTLIE EPL v2 docs

See `docs/SOURCES.md` for the cited links and key takeaways.


## Python requirement

This project requires Python 3.10 or newer. Current `penaltyblog` releases on PyPI require Python >=3.10.
For macOS, Python 3.11 is a good default.

```bash
brew install python@3.11
python3.11 --version
```


## macOS path-with-spaces note

When using a config path like `/Users/.../EPL Page/...`, always quote it. The Makefile in this build already does that for you.
