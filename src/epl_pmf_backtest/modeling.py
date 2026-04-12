from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from .calibration import fit_grid_temperature, temperature_scale_grid
from .markets import clip_grid, goal_expectations, grid_log_loss, normalize_grid, one_x_two_from_grid


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



def _to_writable_numeric_array(series: pd.Series, dtype: Any) -> np.ndarray:
    arr = np.array(series.to_numpy(copy=True), dtype=dtype, copy=True)
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _to_writable_string_array(series: pd.Series) -> np.ndarray:
    arr = np.array(series.astype(str).to_numpy(copy=True), dtype=object, copy=True)
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _fit_single_model(name: str, train_df: pd.DataFrame, xi: float) -> Any:
    pb = _require_penaltyblog()
    class_name = MODEL_CLASS_NAMES[name]
    model_cls = getattr(pb.models, class_name)

    goals_home = _to_writable_numeric_array(train_df["goals_home"].astype(int), np.int64)
    goals_away = _to_writable_numeric_array(train_df["goals_away"].astype(int), np.int64)
    team_home = _to_writable_string_array(train_df["team_home"])
    team_away = _to_writable_string_array(train_df["team_away"])
    weights = np.array(build_time_weights(pd.to_datetime(train_df["date"]), xi), dtype=np.float64, copy=True)
    weights = np.ascontiguousarray(weights)
    if not weights.flags.writeable:
        weights = weights.copy()

    model = model_cls(
        goals_home,
        goals_away,
        team_home,
        team_away,
        weights,
    )
    model.fit()
    return model


def _predict_model_grid(model: Any, home_team: str, away_team: str, max_goals: int) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        probs = model.predict(home_team, away_team)
    except ValueError as e:
        msg = str(e)
        if "Both teams must have been in the training data." in msg:
            raise RuntimeError(
                f"UNSEEN_TEAM_IN_FOLD: {home_team} vs {away_team}. "
                "Fail-closed is enabled. No synthetic fallback is allowed."
            ) from e
        raise
    grid, meta = _extract_joint_pmf(probs, max_goals)
    home_exp, away_exp = goal_expectations(grid)
    meta.update({
        "home_goal_expectation": home_exp,
        "away_goal_expectation": away_exp,
        "home_draw_away": one_x_two_from_grid(grid),
    })
    return grid, meta


def _pi_prior_for_fixture(train_df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, float]:
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
) -> tuple[Dict[str, Any], EnsembleFit]:
    fitted_models = {name: _fit_single_model(name, train_df, time_decay_xi) for name in model_names}

    per_model_val_losses = {name: [] for name in model_names}
    val_predictions: Dict[str, List[np.ndarray]] = {name: [] for name in model_names}
    val_pi_priors: List[Dict[str, float]] = []
    val_outcomes: List[Tuple[int, int]] = []

    for row in validation_df.itertuples(index=False):
        home = str(row.team_home)
        away = str(row.team_away)
        for name, model in fitted_models.items():
            grid, _ = _predict_model_grid(model, home, away, max_goals)
            val_predictions[name].append(grid)
            per_model_val_losses[name].append(grid_log_loss(grid, int(row.goals_home), int(row.goals_away)))
        val_pi_priors.append(_pi_prior_for_fixture(train_df, home, away))
        val_outcomes.append((int(row.goals_home), int(row.goals_away)))

    avg_losses = {name: float(np.mean(losses)) for name, losses in per_model_val_losses.items()}
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

    best_pi_blend = 0.0
    best_temperature = 1.0
    best_loss = float("inf")
    for pi_blend in pi_blend_grid:
        tilted = [
            blend_grids(grid, tilt_grid_to_1x2(grid, pi_prior), float(pi_blend))
            for grid, pi_prior in zip(raw_ensemble_grids, val_pi_priors)
        ]
        temp, loss = fit_grid_temperature(tilted, val_outcomes, calibration_temperatures)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temp
            best_pi_blend = float(pi_blend)

    diagnostics = {
        "validation_model_log_loss": avg_losses,
        "validation_ensemble_log_loss": best_loss,
        "ensemble_softmax_temperature": ensemble_softmax_temperature,
    }
    return fitted_models, EnsembleFit(
        model_names=list(model_names),
        model_weights=model_weights,
        calibration_temperature=float(best_temperature),
        pi_blend_weight=float(best_pi_blend),
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
) -> Dict[str, Any]:
    component_grids: Dict[str, np.ndarray] = {}
    component_meta: Dict[str, Any] = {}
    for name, model in fitted_models.items():
        grid, meta = _predict_model_grid(model, home_team, away_team, max_goals)
        component_grids[name] = grid
        component_meta[name] = meta

    ensemble_grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for name, weight in ensemble_fit.model_weights.items():
        ensemble_grid += weight * component_grids[name]
    ensemble_grid = clip_grid(ensemble_grid)

    pi_prior = _pi_prior_for_fixture(train_df_for_pi, home_team, away_team)
    tilted_grid = tilt_grid_to_1x2(ensemble_grid, pi_prior)
    blended = blend_grids(ensemble_grid, tilted_grid, ensemble_fit.pi_blend_weight)
    calibrated = temperature_scale_grid(blended, ensemble_fit.calibration_temperature)

    home_exp, away_exp = goal_expectations(calibrated)
    return {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_utc": kickoff_utc,
        "joint_pmf": calibrated.tolist(),
        "goal_expectations": {"home": home_exp, "away": away_exp},
        "raw_1x2": one_x_two_from_grid(ensemble_grid),
        "pi_prior_1x2": pi_prior,
        "final_1x2": one_x_two_from_grid(calibrated),
        "ensemble_weights": ensemble_fit.model_weights,
        "calibration_temperature": ensemble_fit.calibration_temperature,
        "pi_blend_weight": ensemble_fit.pi_blend_weight,
        "component_models": component_meta,
    }
