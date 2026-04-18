from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import uuid

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .features import build_feature_table
from .markets import (
    derived_markets,
    grid_log_loss,
    one_x_two_from_grid,
    select_value_bets,
    three_way_implied_probs_decimal,
)
from .modeling import fit_ensemble, predict_with_ensemble
from .providers.historical_football_data import HistoricalFootballDataProvider
from .storage import ensure_dirs, write_csv, write_json


def make_walkforward_windows(
    n_matches: int,
    train_min_matches: int,
    validation_matches: int,
    test_step_matches: int,
) -> List[Tuple[slice, slice, slice]]:
    windows = []
    start_test = train_min_matches + validation_matches
    fold_id = 0
    while start_test < n_matches:
        train_slice = slice(0, start_test - validation_matches)
        val_slice = slice(start_test - validation_matches, start_test)
        test_slice = slice(start_test, min(start_test + test_step_matches, n_matches))
        windows.append((train_slice, val_slice, test_slice))
        start_test += test_step_matches
        fold_id += 1
    return windows


def _summarize_fold(predictions: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "n_matches": int(len(predictions)),
        "joint_log_loss": float(predictions["joint_log_loss"].mean()),
        "home_brier": float(np.mean((predictions["model_home_win"] - predictions["actual_home_win"]) ** 2)),
        "draw_brier": float(np.mean((predictions["model_draw"] - predictions["actual_draw"]) ** 2)),
        "away_brier": float(np.mean((predictions["model_away_win"] - predictions["actual_away_win"]) ** 2)),
    }
    betting_enabled = True
    if "__betting_enabled__" in predictions.columns and len(predictions):
        betting_enabled = bool(predictions["__betting_enabled__"].iloc[0])

    if betting_enabled and "bet_edge" in predictions and predictions["bet_edge"].notna().any():
        bet_mask = predictions["bet_edge"].notna()
        out["n_bets"] = int(bet_mask.sum())
        out["avg_bet_edge"] = float(predictions.loc[bet_mask, "bet_edge"].mean())
        out["avg_stake_fraction"] = float(predictions.loc[bet_mask, "bet_stake_fraction"].mean())
    else:
        out["n_bets"] = 0
        out["avg_bet_edge"] = 0.0
        out["avg_stake_fraction"] = 0.0
    return out


def sync_historical(config: ProjectConfig) -> Path:
    ensure_dirs(config.storage_root)
    historical = config.raw.get("providers", {}).get("historical", {})
    provider = HistoricalFootballDataProvider(
        league=config.league,
        seasons=historical.get("seasons", []),
        odds_priority=historical.get("odds_priority", ["ps", "p", "max", "avg", "b365"]),
        storage_root=config.storage_root,
    )
    df = provider.fetch(use_cache=False)
    out_path = config.storage_root / "curated" / "historical_matches.csv"
    write_csv(out_path, df)

    priors_cfg = config.raw.get("providers", {}).get("promoted_priors", {})
    if priors_cfg.get("enabled", False):
        lower_provider = HistoricalFootballDataProvider(
            league=priors_cfg.get("lower_division_league", "ENG Championship"),
            seasons=priors_cfg.get("seasons", historical.get("seasons", [])),
            odds_priority=[],
            storage_root=config.storage_root,
        )
        lower_df = lower_provider.fetch(use_cache=False)
        lower_path = config.storage_root / "curated" / "lower_division_matches.csv"
        write_csv(lower_path, lower_df)
    return out_path



def _force_disable_bets(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    for col in ["bet_market", "bet_selection"]:
        if col in predictions.columns:
            predictions[col] = None
    for col in ["bet_edge", "bet_stake_fraction", "bet_decimal_odds", "bet_result_profit_per_unit"]:
        if col in predictions.columns:
            predictions[col] = np.nan
    predictions["__betting_enabled__"] = False
    return predictions

def run_backtest(config: ProjectConfig) -> Path:
    ensure_dirs(config.storage_root)

    historical = config.raw.get("providers", {}).get("historical", {})
    provider = HistoricalFootballDataProvider(
        league=config.league,
        seasons=historical.get("seasons", []),
        odds_priority=historical.get("odds_priority", ["ps", "p", "max", "avg", "b365"]),
        storage_root=config.storage_root,
    )
    matches = provider.fetch(use_cache=True)
    priors_cfg = config.raw.get("providers", {}).get("promoted_priors", {})
    lower_division_df = None
    if priors_cfg.get("enabled", False):
        lower_provider = HistoricalFootballDataProvider(
            league=priors_cfg.get("lower_division_league", "ENG Championship"),
            seasons=priors_cfg.get("seasons", historical.get("seasons", [])),
            odds_priority=[],
            storage_root=config.storage_root,
        )
        lower_division_df = lower_provider.fetch(use_cache=True)
        write_csv(config.storage_root / "curated" / "lower_division_matches.csv", lower_division_df)

    features = build_feature_table(matches)
    features_path = config.storage_root / "features" / "historical_features.csv"
    write_csv(features_path, features)

    wf = config.section("walkforward")
    model_cfg = config.section("model")
    betting_cfg = config.section("betting")
    betting_enabled = bool(betting_cfg.get("enabled", True)) and bool(betting_cfg.get("markets", []))
 

    windows = make_walkforward_windows(
        n_matches=len(features),
        train_min_matches=int(wf.get("train_min_matches", 760)),
        validation_matches=int(wf.get("validation_matches", 190)),
        test_step_matches=int(wf.get("test_step_matches", 50)),
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    all_rows: List[Dict[str, Any]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold_id, (train_slice, val_slice, test_slice) in enumerate(windows):
        train_df = features.iloc[train_slice].copy()
        val_df = features.iloc[val_slice].copy()
        test_df = features.iloc[test_slice].copy()

        fitted_models, ensemble_fit = fit_ensemble(
            train_df=train_df,
            validation_df=val_df,
            model_names=model_cfg.get("base_models", []),
            max_goals=int(model_cfg.get("max_goals", 8)),
            time_decay_xi=float(model_cfg.get("time_decay_xi", 0.001)),
            ensemble_softmax_temperature=float(model_cfg.get("ensemble_softmax_temperature", 0.5)),
            calibration_temperatures=model_cfg.get("calibration_temperatures", [1.0]),
            pi_blend_grid=model_cfg.get("pi_blend_grid", [0.0]),
            lower_division_df=lower_division_df,
            top_league=str(config.league),
            lower_division_league=str(priors_cfg.get("lower_division_league", "ENG Championship")),
            prior_matches_for_transition=int(priors_cfg.get("prior_matches_for_transition", 10)),
            strength_shrink_matches=float(priors_cfg.get("strength_shrink_matches", 8.0)),
        )

        fold_rows: List[Dict[str, Any]] = []
        for row in test_df.itertuples(index=False):
            forecast = predict_with_ensemble(
                fitted_models=fitted_models,
                ensemble_fit=ensemble_fit,
                train_df_for_pi=train_df,
                home_team=str(row.team_home),
                away_team=str(row.team_away),
                kickoff_utc=pd.Timestamp(row.date).isoformat(),
                match_id=str(row.match_id),
                max_goals=int(model_cfg.get("max_goals", 8)),
                match_season=str(row.season),
                lower_division_df=lower_division_df,
                market_odds={
                    "odds_home": float(row.odds_home) if pd.notna(row.odds_home) else None,
                    "odds_draw": float(row.odds_draw) if pd.notna(row.odds_draw) else None,
                    "odds_away": float(row.odds_away) if pd.notna(row.odds_away) else None,
                },
                top_league=str(config.league),
                lower_division_league=str(priors_cfg.get("lower_division_league", "ENG Championship")),
                prior_matches_for_transition=int(priors_cfg.get("prior_matches_for_transition", 10)),
                strength_shrink_matches=float(priors_cfg.get("strength_shrink_matches", 8.0)),
            )

            grid = np.asarray(forecast["joint_pmf"], dtype=float)
            one_x_two = one_x_two_from_grid(grid)
            actual_home = int(row.goals_home)
            actual_away = int(row.goals_away)

            record: Dict[str, Any] = {
                "run_id": run_id,
                "fold_id": fold_id,
                "match_id": row.match_id,
                "date": pd.Timestamp(row.date).isoformat(),
                "season": row.season,
                "competition": row.competition,
                "home_team": row.team_home,
                "away_team": row.team_away,
                "goals_home": actual_home,
                "goals_away": actual_away,
                "joint_log_loss": grid_log_loss(grid, actual_home, actual_away),
                "model_home_win": one_x_two["home"],
                "model_draw": one_x_two["draw"],
                "model_away_win": one_x_two["away"],
                "actual_home_win": float(actual_home > actual_away),
                "actual_draw": float(actual_home == actual_away),
                "actual_away_win": float(actual_home < actual_away),
                "market_odds_home": row.odds_home,
                "market_odds_draw": row.odds_draw,
                "market_odds_away": row.odds_away,
                "ensemble_weights_json": forecast["ensemble_weights"],
                "calibration_temperature": forecast["calibration_temperature"],
                "pi_blend_weight": forecast["pi_blend_weight"],
                "joint_pmf_json": forecast["joint_pmf"],
                "derived_markets_json": derived_markets(grid),
                "goal_expectations_home": forecast["goal_expectations"]["home"],
                "goal_expectations_away": forecast["goal_expectations"]["away"],
                "pi_home_win": forecast["pi_prior_1x2"]["home"],
                "pi_draw": forecast["pi_prior_1x2"]["draw"],
                "pi_away_win": forecast["pi_prior_1x2"]["away"],
                "market_regime": forecast.get("market_regime"),
                "market_blend_weight": forecast.get("market_blend_weight"),
                "forecast_source": forecast.get("forecast_source", "ensemble"),
            }

            if pd.notna(row.odds_home) and pd.notna(row.odds_draw) and pd.notna(row.odds_away):
                fair = three_way_implied_probs_decimal(
                    float(row.odds_home),
                    float(row.odds_draw),
                    float(row.odds_away),
                    method=str(betting_cfg.get("no_vig_method", "proportional")),
                )
                record["market_fair_home"] = fair["home"]
                record["market_fair_draw"] = fair["draw"]
                record["market_fair_away"] = fair["away"]
                bets = select_value_bets(
                    grid=grid,
                    odds_row={
                        "odds_home": float(row.odds_home),
                        "odds_draw": float(row.odds_draw),
                        "odds_away": float(row.odds_away),
                    },
                    min_edge=float(betting_cfg.get("min_edge", 0.025)),
                    min_decimal_odds=float(betting_cfg.get("min_decimal_odds", 1.20)),
                    max_decimal_odds=float(betting_cfg.get("max_decimal_odds", 10.0)),
                    kelly_fraction=float(betting_cfg.get("kelly_fraction", 0.2)),
                )
                if betting_enabled:
                    if bets:
                        best = bets[0]
                        record["bet_market"] = best["market"]
                        record["bet_selection"] = best["selection"]
                        record["bet_edge"] = best["edge"]
                        record["bet_stake_fraction"] = best["stake_fraction"]
                        record["bet_decimal_odds"] = best["decimal_odds"]
                        record["bet_result_profit_per_unit"] = (
                            best["decimal_odds"] - 1.0
                            if (
                                (best["selection"] == "home" and actual_home > actual_away)
                                or (best["selection"] == "draw" and actual_home == actual_away)
                                or (best["selection"] == "away" and actual_home < actual_away)
                            )
                            else -1.0
                        )
                    else:
                        record["bet_market"] = None
                        record["bet_selection"] = None
                        record["bet_edge"] = np.nan
                        record["bet_stake_fraction"] = np.nan
                        record["bet_decimal_odds"] = np.nan
                        record["bet_result_profit_per_unit"] = np.nan
                else:
                    record["bet_market"] = None
                    record["bet_selection"] = None
                    record["bet_edge"] = np.nan
                    record["bet_stake_fraction"] = np.nan
                    record["bet_decimal_odds"] = np.nan
                    record["bet_result_profit_per_unit"] = np.nan
            fold_rows.append(record)

        fold_df = pd.DataFrame(fold_rows)
        all_rows.extend(fold_rows)
        fold_summary = _summarize_fold(fold_df)
        fold_summary.update({
            "fold_id": fold_id,
            "train_start": pd.Timestamp(train_df["date"].min()).isoformat(),
            "train_end": pd.Timestamp(train_df["date"].max()).isoformat(),
            "validation_start": pd.Timestamp(val_df["date"].min()).isoformat(),
            "validation_end": pd.Timestamp(val_df["date"].max()).isoformat(),
            "test_start": pd.Timestamp(test_df["date"].min()).isoformat(),
            "test_end": pd.Timestamp(test_df["date"].max()).isoformat(),
            "ensemble_weights": ensemble_fit.model_weights,
            "calibration_temperature": ensemble_fit.calibration_temperature,
            "pi_blend_weight": ensemble_fit.pi_blend_weight,
            "diagnostics": ensemble_fit.diagnostics,
        })
        fold_summaries.append(fold_summary)

    predictions_df = pd.DataFrame(all_rows)
    if not betting_enabled:
        for col in ["bet_market", "bet_selection"]:
            if col in predictions_df.columns:
                predictions_df[col] = None
        for col in ["bet_edge", "bet_stake_fraction", "bet_decimal_odds", "bet_result_profit_per_unit"]:
            if col in predictions_df.columns:
                predictions_df[col] = np.nan
    predictions_path = config.storage_root / "backtests" / f"{run_id}_predictions.csv"
    folds_path = config.storage_root / "backtests" / f"{run_id}_folds.json"
    summary_path = config.storage_root / "backtests" / f"{run_id}_summary.json"

    write_csv(predictions_path, predictions_df)
    write_json(folds_path, fold_summaries)

    overall = _summarize_fold(predictions_df)
    overall.update({
        "run_id": run_id,
        "n_folds": len(fold_summaries),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
    })
    write_json(summary_path, overall)
    write_json(config.storage_root / "backtests" / "latest_run.json", {
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "folds_path": str(folds_path),
        "summary_path": str(summary_path),
    })
    return summary_path
