from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from .calibration import fit_grid_temperature, temperature_scale_grid
from .markets import clip_grid, goal_expectations, grid_log_loss, normalize_grid, one_x_two_from_grid
from .promoted_priors import PromotedPriorContext, build_promoted_prior_context, build_real_data_prior_grid


MODEL_CLASS_NAMES = {
    "dixon_coles": "DixonColesGoalModel",
    "bivariate_poisson": "BivariatePoissonGoalModel",
    "zero_inflated_poisson": "ZeroInflatedPoissonGoalsModel",
    "negative_binomial": "NegativeBinomialGoalModel",
    "weibull_copula": "WeibullCopulaGoalsModel",
}


@dataclass(slots=True)
class EnsembleFit:
    model_names: List[str]
    model_weights: Dict[str, float]
    calibration_temperature: float
    pi_blend_weight: float
    market_regime_blend_weights: Dict[str, float]
    diagnostics: Dict[str, Any]


def _require_penaltyblog():
    try:
        import penaltyblog as pb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("penaltyblog is required for modeling") from exc
    return pb


def build_time_weights(dates: pd.Series, xi: float) -> np.ndarray:
    pb = _require_penaltyblog()
    return np.asarray(pb.models.dixon_coles_weights(dates, xi), dtype=float)


def _safe_goal_expectation(prob_grid: Any, side: str) -> float | None:
    candidates = [
        f"{side}_goal_expectation",
        f"{side}_goals_expectation",
        f"{side}_expectation",
        f"expected_{side}_goals",
    ]
    for name in candidates:
        value = getattr(prob_grid, name, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > 0:
                return float(np.asarray(value).reshape(-1)[0])
        try:
            return float(value)
        except Exception:
            continue
    return None


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    lam = max(float(lam), 1e-9)
    return np.exp(-lam) * np.power(lam, k) / np.vectorize(math.factorial)(k)


def _extract_joint_pmf(prob_grid: Any, max_goals: int) -> tuple[np.ndarray, Dict[str, Any]]:
    candidates = ["grid", "probability_grid", "matrix", "probabilities", "joint_pmf"]
    for name in candidates:
        value = getattr(prob_grid, name, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
            arr = arr[: max_goals + 1, : max_goals + 1]
            return clip_grid(arr), {"extractor": f"attribute:{name}"}

    method_candidates = ["exact_score", "score_probability", "correct_score", "probability"]
    for method_name in method_candidates:
        fn = getattr(prob_grid, method_name, None)
        if callable(fn):
            grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
            ok = True
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    try:
                        grid[i, j] = float(fn(i, j))
                    except Exception:
                        ok = False
                        break
                if not ok:
                    break
            if ok and grid.sum() > 0:
                return clip_grid(grid), {"extractor": f"method:{method_name}"}

    home_exp = _safe_goal_expectation(prob_grid, "home")
    away_exp = _safe_goal_expectation(prob_grid, "away")
    if home_exp is None or away_exp is None:
        raise RuntimeError(
            "Could not extract a joint PMF from Penaltyblog's FootballProbabilityGrid. "
            "Inspect the object in your local environment and extend _extract_joint_pmf."
        )

    k = np.arange(max_goals + 1)
    home_pmf = _poisson_pmf(k, home_exp)
    away_pmf = _poisson_pmf(k, away_exp)
    grid = np.outer(home_pmf, away_pmf)
    return clip_grid(grid), {
        "extractor": "fallback:independent_poisson_from_expectations",
        "home_goal_expectation": home_exp,
        "away_goal_expectation": away_exp,
    }


def _fit_single_model(name: str, train_df: pd.DataFrame, xi: float) -> Any:
    pb = _require_penaltyblog()
    class_name = MODEL_CLASS_NAMES[name]
    model_cls = getattr(pb.models, class_name)

    goals_home = np.ascontiguousarray(train_df["goals_home"].to_numpy(dtype=np.int64, copy=True))
    goals_away = np.ascontiguousarray(train_df["goals_away"].to_numpy(dtype=np.int64, copy=True))
    team_home = np.ascontiguousarray(train_df["team_home"].astype(str).to_numpy(copy=True))
    team_away = np.ascontiguousarray(train_df["team_away"].astype(str).to_numpy(copy=True))
    weights = np.ascontiguousarray(np.array(build_time_weights(train_df["date"], xi), dtype=float, copy=True))

    model = model_cls(
        goals_home,
        goals_away,
        team_home,
        team_away,
        weights,
    )
    model.fit()
    return model



def _predict_model_grid(
    model: Any,
    home_team: str,
    away_team: str,
    max_goals: int,
    *,
    train_df: pd.DataFrame | None = None,
    lower_division_df: pd.DataFrame | None = None,
    match_season: str | None = None,
    promoted_prior_context: PromotedPriorContext | None = None,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        probs = model.predict(home_team, away_team)
        grid, meta = _extract_joint_pmf(probs, max_goals)
    except ValueError as e:
        msg = str(e)
        if "Both teams must have been in the training data." not in msg:
            raise
        if train_df is None or match_season is None:
            raise RuntimeError(
                f"UNSEEN_TEAM_IN_FOLD: {home_team} vs {away_team}. "
                "No training dataframe or match season was supplied to build a real promoted-team prior."
            ) from e
        grid, meta = build_real_data_prior_grid(
            train_df=train_df,
            lower_division_df=lower_division_df,
            home_team=home_team,
            away_team=away_team,
            match_season=str(match_season),
            max_goals=max_goals,
            top_league=top_league,
            lower_division_league=lower_division_league,
            context=promoted_prior_context,
        )
        meta["fallback_reason"] = "unseen_team_in_top_league_training_fold"
    home_exp, away_exp = goal_expectations(grid)
    meta.update({
        "home_goal_expectation": home_exp,
        "away_goal_expectation": away_exp,
        "home_draw_away": one_x_two_from_grid(grid),
    })
    return grid, meta



def _pi_prior_for_fixture(
    train_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    lower_division_df: pd.DataFrame | None = None,
    match_season: str | None = None,
    promoted_prior_context: PromotedPriorContext | None = None,
    max_goals: int = 8,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
) -> Dict[str, float]:
    seen_teams = set(train_df["team_home"]).union(set(train_df["team_away"]))
    if home_team not in seen_teams or away_team not in seen_teams:
        if match_season is None:
            raise RuntimeError(
                f"UNSEEN_TEAM_IN_FOLD: {home_team} vs {away_team}. "
                "Cannot build prior probabilities without the match season."
            )
        prior_grid, _ = build_real_data_prior_grid(
            train_df=train_df,
            lower_division_df=lower_division_df,
            home_team=home_team,
            away_team=away_team,
            match_season=str(match_season),
            max_goals=max_goals,
            top_league=top_league,
            lower_division_league=lower_division_league,
            context=promoted_prior_context,
        )
        return one_x_two_from_grid(prior_grid)

    pb = _require_penaltyblog()
    pi = pb.ratings.PiRatingSystem()
    ordered = train_df.sort_values("date")
    for row in ordered.itertuples(index=False):
        goal_diff = int(row.goals_home) - int(row.goals_away)
        pi.update_ratings(str(row.team_home), str(row.team_away), goal_diff)
    probs = pi.calculate_match_probabilities(home_team, away_team)
    return {
        "home": float(probs["home_win"]),
        "draw": float(probs["draw"]),
        "away": float(probs["away_win"]),
    }


def _market_prior_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> Dict[str, float]:
    from epl_pmf_backtest.markets import three_way_implied_probs_decimal

    fair = three_way_implied_probs_decimal(
        float(odds_home),
        float(odds_draw),
        float(odds_away),
        method="proportional",
    )
    return {
        "home": float(fair["home"]),
        "draw": float(fair["draw"]),
        "away": float(fair["away"]),
    }


def _market_regime_from_prior(market_prior: Dict[str, float]) -> str:
    diff = float(market_prior["home"]) - float(market_prior["away"])
    if diff >= 0.18:
        return "home_fav"
    if diff <= -0.18:
        return "away_fav"
    return "balanced"


def tilt_grid_to_1x2(grid: np.ndarray, target_probs: Dict[str, float]) -> np.ndarray:
    out = np.array(grid, dtype=float, copy=True)
    current = one_x_two_from_grid(out)
    block_weights = {}
    for key in ("home", "draw", "away"):
        if current[key] <= 0:
            block_weights[key] = 1.0
        else:
            block_weights[key] = target_probs[key] / current[key]

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if i > j:
                out[i, j] *= block_weights["home"]
            elif i == j:
                out[i, j] *= block_weights["draw"]
            else:
                out[i, j] *= block_weights["away"]
    return clip_grid(out)


def blend_grids(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    return clip_grid((1.0 - weight_b) * a + weight_b * b)



def fit_ensemble(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model_names: Sequence[str],
    max_goals: int,
    time_decay_xi: float,
    ensemble_softmax_temperature: float,
    calibration_temperatures: Iterable[float],
    pi_blend_grid: Iterable[float],
    *,
    lower_division_df: pd.DataFrame | None = None,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
    prior_matches_for_transition: int = 10,
    strength_shrink_matches: float = 8.0,
) -> tuple[Dict[str, Any], EnsembleFit]:
    fitted_models = {name: _fit_single_model(name, train_df, time_decay_xi) for name in model_names}
    promoted_prior_context = build_promoted_prior_context(
        train_df,
        lower_division_df,
        top_league=top_league,
        lower_division_league=lower_division_league,
        prior_matches_for_transition=prior_matches_for_transition,
        strength_shrink_matches=strength_shrink_matches,
    )

    seen_teams = set(train_df["team_home"]).union(set(train_df["team_away"]))
    per_model_val_losses = {name: [] for name in model_names}
    val_predictions: Dict[str, List[np.ndarray]] = {name: [] for name in model_names}
    val_pi_priors: List[Dict[str, float]] = []
    val_market_priors: List[Dict[str, float]] = []
    val_market_regimes: List[str] = []
    val_outcomes: List[Tuple[int, int]] = []

    for row in validation_df.itertuples(index=False):
        home = str(row.team_home)
        away = str(row.team_away)
        if pd.notna(row.odds_home) and pd.notna(row.odds_draw) and pd.notna(row.odds_away):
            market_prior = _market_prior_from_odds(float(row.odds_home), float(row.odds_draw), float(row.odds_away))
        else:
            market_prior = {"home": 1.0 / 3.0, "draw": 1.0 / 3.0, "away": 1.0 / 3.0}
        market_regime = _market_regime_from_prior(market_prior)
        needs_promoted_prior = home not in seen_teams or away not in seen_teams

        if needs_promoted_prior:
            prior_grid, prior_meta = build_real_data_prior_grid(
                train_df=train_df,
                lower_division_df=lower_division_df,
                home_team=home,
                away_team=away,
                match_season=str(row.season),
                max_goals=max_goals,
                top_league=top_league,
                lower_division_league=lower_division_league,
                context=promoted_prior_context,
            )
            for name in model_names:
                val_predictions[name].append(prior_grid)
            val_pi_priors.append(one_x_two_from_grid(prior_grid))
            val_market_priors.append(market_prior)
            val_market_regimes.append(market_regime)
            val_outcomes.append((int(row.goals_home), int(row.goals_away)))
            continue

        for name, model in fitted_models.items():
            grid, _ = _predict_model_grid(
                model,
                home,
                away,
                max_goals,
                train_df=train_df,
                lower_division_df=lower_division_df,
                match_season=str(row.season),
                promoted_prior_context=promoted_prior_context,
                top_league=top_league,
                lower_division_league=lower_division_league,
            )
            val_predictions[name].append(grid)
            per_model_val_losses[name].append(grid_log_loss(grid, int(row.goals_home), int(row.goals_away)))
        val_pi_priors.append(_pi_prior_for_fixture(
            train_df,
            home,
            away,
            lower_division_df=lower_division_df,
            match_season=str(row.season),
            promoted_prior_context=promoted_prior_context,
            max_goals=max_goals,
            top_league=top_league,
            lower_division_league=lower_division_league,
        ))
        val_market_priors.append(market_prior)
        val_market_regimes.append(market_regime)
        val_outcomes.append((int(row.goals_home), int(row.goals_away)))

    avg_losses = {
        name: float(np.mean(losses)) if len(losses) else 999.0
        for name, losses in per_model_val_losses.items()
    }
    logits = np.array([-avg_losses[name] / max(ensemble_softmax_temperature, 1e-9) for name in model_names], dtype=float)
    logits -= logits.max()
    raw_weights = np.exp(logits)
    raw_weights /= raw_weights.sum()
    model_weights = {name: float(weight) for name, weight in zip(model_names, raw_weights)}

    raw_ensemble_grids: List[np.ndarray] = []
    for i in range(len(validation_df)):
        grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for name in model_names:
            grid += model_weights[name] * val_predictions[name][i]
        raw_ensemble_grids.append(clip_grid(grid))

    market_blend_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_market_weights = {"home_fav": 0.0, "away_fav": 0.0, "balanced": 0.0}
    best_pi_blend = 0.0
    best_temperature = 1.0
    best_loss = float("inf")
    for w_home in market_blend_grid:
        for w_away in market_blend_grid:
            for w_bal in market_blend_grid:
                market_weights = {
                    "home_fav": float(w_home),
                    "away_fav": float(w_away),
                    "balanced": float(w_bal),
                }
                market_adjusted = []
                for grid, market_prior, regime in zip(raw_ensemble_grids, val_market_priors, val_market_regimes):
                    market_tilted = tilt_grid_to_1x2(grid, market_prior)
                    market_adjusted.append(blend_grids(grid, market_tilted, market_weights[regime]))

                for pi_blend in pi_blend_grid:
                    tilted = [
                        blend_grids(grid, tilt_grid_to_1x2(grid, pi_prior), float(pi_blend))
                        for grid, pi_prior in zip(market_adjusted, val_pi_priors)
                    ]
                    temp, loss = fit_grid_temperature(tilted, val_outcomes, calibration_temperatures)
                    if loss < best_loss:
                        best_loss = loss
                        best_temperature = temp
                        best_pi_blend = float(pi_blend)
                        best_market_weights = market_weights.copy()

    diagnostics = {
        "validation_model_log_loss": avg_losses,
        "validation_ensemble_log_loss": best_loss,
        "ensemble_softmax_temperature": ensemble_softmax_temperature,
        "market_regime_blend_weights": best_market_weights,
    }
    return fitted_models, EnsembleFit(
        model_names=list(model_names),
        model_weights=model_weights,
        calibration_temperature=float(best_temperature),
        pi_blend_weight=float(best_pi_blend),
        market_regime_blend_weights=best_market_weights,
        diagnostics=diagnostics,
    )



def predict_with_ensemble(
    fitted_models: Dict[str, Any],
    ensemble_fit: EnsembleFit,
    train_df_for_pi: pd.DataFrame,
    home_team: str,
    away_team: str,
    kickoff_utc: str,
    match_id: str,
    max_goals: int,
    *,
    match_season: str | None = None,
    lower_division_df: pd.DataFrame | None = None,
    market_odds: Dict[str, float | None] | None = None,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
    prior_matches_for_transition: int = 10,
    strength_shrink_matches: float = 8.0,
) -> Dict[str, Any]:
    promoted_prior_context = build_promoted_prior_context(
        train_df_for_pi,
        lower_division_df,
        top_league=top_league,
        lower_division_league=lower_division_league,
        prior_matches_for_transition=prior_matches_for_transition,
        strength_shrink_matches=strength_shrink_matches,
    )
    component_grids: Dict[str, np.ndarray] = {}
    component_meta: Dict[str, Any] = {}
    for name, model in fitted_models.items():
        grid, meta = _predict_model_grid(
            model,
            home_team,
            away_team,
            max_goals,
            train_df=train_df_for_pi,
            lower_division_df=lower_division_df,
            match_season=match_season,
            promoted_prior_context=promoted_prior_context,
            top_league=top_league,
            lower_division_league=lower_division_league,
        )
        component_grids[name] = grid
        component_meta[name] = meta

    ensemble_grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for name, weight in ensemble_fit.model_weights.items():
        ensemble_grid += weight * component_grids[name]
    ensemble_grid = clip_grid(ensemble_grid)

    market_prior = {"home": 1.0 / 3.0, "draw": 1.0 / 3.0, "away": 1.0 / 3.0}
    market_regime = "balanced"
    market_blend_weight = 0.0
    market_adjusted_grid = ensemble_grid
    if market_odds is not None:
        if all(market_odds.get(k) is not None for k in ["odds_home", "odds_draw", "odds_away"]):
            market_prior = _market_prior_from_odds(
                float(market_odds["odds_home"]),
                float(market_odds["odds_draw"]),
                float(market_odds["odds_away"]),
            )
            market_regime = _market_regime_from_prior(market_prior)
            market_blend_weight = float(ensemble_fit.market_regime_blend_weights.get(market_regime, 0.0))
            market_tilted = tilt_grid_to_1x2(ensemble_grid, market_prior)
            market_adjusted_grid = blend_grids(ensemble_grid, market_tilted, market_blend_weight)

    pi_prior = _pi_prior_for_fixture(
        train_df_for_pi,
        home_team,
        away_team,
        lower_division_df=lower_division_df,
        match_season=match_season,
        promoted_prior_context=promoted_prior_context,
        max_goals=max_goals,
        top_league=top_league,
        lower_division_league=lower_division_league,
    )
    tilted_grid = tilt_grid_to_1x2(market_adjusted_grid, pi_prior)
    blended = blend_grids(market_adjusted_grid, tilted_grid, ensemble_fit.pi_blend_weight)
    calibrated = temperature_scale_grid(blended, ensemble_fit.calibration_temperature)

    if market_odds is not None:
        if all(market_odds.get(k) is not None for k in ["odds_home", "odds_draw", "odds_away"]):
            calibrated = tilt_grid_to_1x2(calibrated, market_prior)

    home_exp, away_exp = goal_expectations(calibrated)
    return {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_utc": kickoff_utc,
        "joint_pmf": calibrated.tolist(),
        "goal_expectations": {"home": home_exp, "away": away_exp},
        "raw_1x2": one_x_two_from_grid(ensemble_grid),
        "market_prior_1x2": market_prior,
        "market_regime": market_regime,
        "market_blend_weight": market_blend_weight,
        "pi_prior_1x2": pi_prior,
        "final_1x2": one_x_two_from_grid(calibrated),
        "ensemble_weights": ensemble_fit.model_weights,
        "calibration_temperature": ensemble_fit.calibration_temperature,
        "pi_blend_weight": ensemble_fit.pi_blend_weight,
        "component_models": component_meta,
        "forecast_source": (
            "promoted_team_prior"
            if any(str(meta.get("source")) == "real_data_promoted_team_prior" for meta in component_meta.values())
            else "ensemble"
        ),
    }
