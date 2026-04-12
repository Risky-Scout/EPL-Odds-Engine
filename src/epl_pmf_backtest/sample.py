from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .reporting import run_report
from .storage import ensure_dirs, write_csv, write_json


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _grid(home_mu: float, away_mu: float, max_goals: int = 8) -> list[list[float]]:
    arr = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            arr[i, j] = _poisson_pmf(i, home_mu) * _poisson_pmf(j, away_mu)
    arr /= arr.sum()
    return arr.tolist()


def build_sample_run(config: ProjectConfig) -> Path:
    ensure_dirs(config.storage_root)
    rng = random.Random(config.random_seed)

    teams = [
        "Arsenal", "Liverpool", "Man City", "Chelsea", "Tottenham",
        "Newcastle", "Brighton", "Aston Villa", "West Ham", "Wolves",
    ]
    rows = []
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    for i in range(150):
        home = teams[i % len(teams)]
        away = teams[(i * 3 + 1) % len(teams)]
        if home == away:
            away = teams[(i * 3 + 2) % len(teams)]
        home_mu = 1.2 + 0.8 * rng.random()
        away_mu = 0.7 + 0.9 * rng.random()
        pmf = _grid(home_mu, away_mu, max_goals=8)
        gh = min(8, np.random.default_rng(i).poisson(home_mu))
        ga = min(8, np.random.default_rng(i + 99).poisson(away_mu))
        one_x_two = {
            "home": float(sum(pmf[x][y] for x in range(9) for y in range(9) if x > y)),
            "draw": float(sum(pmf[x][y] for x in range(9) for y in range(9) if x == y)),
            "away": float(sum(pmf[x][y] for x in range(9) for y in range(9) if x < y)),
        }
        implied_home = max(min(one_x_two["home"] + rng.uniform(-0.04, 0.04), 0.92), 0.05)
        implied_draw = max(min(one_x_two["draw"] + rng.uniform(-0.03, 0.03), 0.40), 0.05)
        implied_away = max(min(1.07 - implied_home - implied_draw, 0.90), 0.05)
        overround = implied_home + implied_draw + implied_away
        market_home = 1 / (implied_home / overround)
        market_draw = 1 / (implied_draw / overround)
        market_away = 1 / (implied_away / overround)

        selection = max(one_x_two, key=one_x_two.get)
        edge = rng.uniform(0.0, 0.08)
        won = (selection == "home" and gh > ga) or (selection == "draw" and gh == ga) or (selection == "away" and gh < ga)
        rows.append({
            "run_id": "sample_run",
            "fold_id": i // 25,
            "match_id": str(10000 + i),
            "date": (start + timedelta(days=i * 2)).isoformat(),
            "season": "2024-2025",
            "competition": "ENG Premier League",
            "home_team": home,
            "away_team": away,
            "goals_home": gh,
            "goals_away": ga,
            "joint_log_loss": -math.log(max(pmf[min(gh, 8)][min(ga, 8)], 1e-12)),
            "model_home_win": one_x_two["home"],
            "model_draw": one_x_two["draw"],
            "model_away_win": one_x_two["away"],
            "actual_home_win": float(gh > ga),
            "actual_draw": float(gh == ga),
            "actual_away_win": float(gh < ga),
            "market_odds_home": market_home,
            "market_odds_draw": market_draw,
            "market_odds_away": market_away,
            "market_fair_home": implied_home / overround,
            "market_fair_draw": implied_draw / overround,
            "market_fair_away": implied_away / overround,
            "bet_market": "1x2",
            "bet_selection": selection,
            "bet_edge": edge if edge > 0.03 else np.nan,
            "bet_stake_fraction": 0.02 if edge > 0.03 else np.nan,
            "bet_decimal_odds": {"home": market_home, "draw": market_draw, "away": market_away}[selection],
            "bet_result_profit_per_unit": (
                {"home": market_home, "draw": market_draw, "away": market_away}[selection] - 1.0 if won else -1.0
            ) if edge > 0.03 else np.nan,
            "ensemble_weights_json": {"dixon_coles": 0.22, "bivariate_poisson": 0.18, "zero_inflated_poisson": 0.12, "negative_binomial": 0.18, "weibull_copula": 0.30},
            "calibration_temperature": 0.9,
            "pi_blend_weight": 0.10,
            "joint_pmf_json": pmf,
            "derived_markets_json": {"1x2": one_x_two},
            "goal_expectations_home": home_mu,
            "goal_expectations_away": away_mu,
            "pi_home_win": one_x_two["home"] * 0.98,
            "pi_draw": one_x_two["draw"] * 1.01,
            "pi_away_win": one_x_two["away"] * 1.01,
        })

    df = pd.DataFrame(rows)
    predictions_path = config.storage_root / "backtests" / "sample_run_predictions.csv"
    summary_path = config.storage_root / "backtests" / "sample_run_summary.json"
    latest_path = config.storage_root / "backtests" / "latest_run.json"

    write_csv(predictions_path, df)
    write_json(summary_path, {
        "run_id": "sample_run",
        "n_folds": int(df["fold_id"].nunique()),
        "n_matches": int(len(df)),
        "joint_log_loss": float(df["joint_log_loss"].mean()),
        "n_bets": int(df["bet_edge"].notna().sum()),
        "avg_bet_edge": float(df["bet_edge"].dropna().mean()),
        "avg_stake_fraction": float(df["bet_stake_fraction"].dropna().mean()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Bundled sample run for report and visual testing. Replace with a real historical walk-forward run.",
    })
    write_json(latest_path, {
        "run_id": "sample_run",
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path),
        "folds_path": "",
    })
    run_report(config)
    return predictions_path
