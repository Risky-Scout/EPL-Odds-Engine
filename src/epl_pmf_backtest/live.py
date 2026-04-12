from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .markets import clip_grid
from .providers.live_bdl import BDLEPLClient


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    factorial = np.vectorize(lambda x: 1 if x == 0 else np.math.factorial(int(x)))
    return np.exp(-lam) * np.power(lam, k) / factorial(k)


def _independent_residual_grid(home_mu: float, away_mu: float, max_goals: int) -> np.ndarray:
    k = np.arange(max_goals + 1)
    home = _poisson_pmf(k, max(home_mu, 1e-9))
    away = _poisson_pmf(k, max(away_mu, 1e-9))
    return clip_grid(np.outer(home, away))


def _extract_live_state_factors(live_state: Dict[str, Any], live_cfg: Dict[str, Any]) -> Dict[str, float]:
    match = live_state.get("match", {}) or {}
    stats = live_state.get("team_match_stats", []) or []
    events = live_state.get("events", []) or []

    minute = float(match.get("minute") or match.get("elapsed") or 0.0)
    home_score = int(match.get("home_score") or match.get("score_home") or 0)
    away_score = int(match.get("away_score") or match.get("score_away") or 0)

    red_home = red_away = 0
    for event in events:
        t = str(event.get("type") or "").lower()
        if "red" in t:
            if event.get("team_id") == match.get("home_team_id"):
                red_home += 1
            elif event.get("team_id") == match.get("away_team_id"):
                red_away += 1

    shots_home = shots_away = 0.0
    poss_home = poss_away = 50.0
    for item in stats:
        team_id = item.get("team_id")
        if team_id == match.get("home_team_id"):
            shots_home = float(item.get("shots") or 0.0)
            poss_home = float(item.get("possession_pct") or 50.0)
        elif team_id == match.get("away_team_id"):
            shots_away = float(item.get("shots") or 0.0)
            poss_away = float(item.get("possession_pct") or 50.0)

    red_card_multiplier = float(live_cfg.get("red_card_multiplier", 0.12))
    shot_pressure_weight = float(live_cfg.get("shot_pressure_weight", 0.01))
    possession_weight = float(live_cfg.get("possession_weight", 0.002))

    home_factor = 1.0
    away_factor = 1.0

    home_factor *= 1.0 + (shots_home - shots_away) * shot_pressure_weight
    away_factor *= 1.0 + (shots_away - shots_home) * shot_pressure_weight

    home_factor *= 1.0 + (poss_home - 50.0) * possession_weight
    away_factor *= 1.0 + (poss_away - 50.0) * possession_weight

    if red_home > red_away:
        home_factor *= 1.0 - (red_home - red_away) * red_card_multiplier
        away_factor *= 1.0 + (red_home - red_away) * red_card_multiplier
    elif red_away > red_home:
        away_factor *= 1.0 - (red_away - red_home) * red_card_multiplier
        home_factor *= 1.0 + (red_away - red_home) * red_card_multiplier

    home_factor = max(home_factor, 0.05)
    away_factor = max(away_factor, 0.05)

    return {
        "minute": minute,
        "home_score": home_score,
        "away_score": away_score,
        "home_factor": home_factor,
        "away_factor": away_factor,
    }


def update_pregame_grid_to_live(
    pregame_grid: np.ndarray,
    live_state: Dict[str, Any],
    live_cfg: Dict[str, Any],
) -> np.ndarray:
    max_goals = pregame_grid.shape[0] - 1
    x = np.arange(max_goals + 1, dtype=float)
    y = np.arange(max_goals + 1, dtype=float)
    pre_home_mu = float(np.sum(pregame_grid * x[:, None]))
    pre_away_mu = float(np.sum(pregame_grid * y[None, :]))

    factors = _extract_live_state_factors(live_state, live_cfg)
    minute = factors["minute"]
    total_minutes = float(live_cfg.get("default_total_match_minutes", 95))
    remaining_fraction = min(max((total_minutes - minute) / total_minutes, 0.0), 1.0)

    residual_home_mu = pre_home_mu * remaining_fraction * factors["home_factor"]
    residual_away_mu = pre_away_mu * remaining_fraction * factors["away_factor"]

    residual = _independent_residual_grid(residual_home_mu, residual_away_mu, max_goals)
    final_grid = np.zeros_like(pregame_grid)
    for dh in range(max_goals + 1):
        for da in range(max_goals + 1):
            fh = min(max_goals, factors["home_score"] + dh)
            fa = min(max_goals, factors["away_score"] + da)
            final_grid[fh, fa] += residual[dh, da]
    return clip_grid(final_grid)


def build_live_snapshot(
    client: BDLEPLClient,
    pregame_forecasts: Iterable[Dict[str, Any]],
    live_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for forecast in pregame_forecasts:
        match_id = int(forecast["match_id"])
        live_state = client.build_live_state(match_id)
        pregame_grid = np.asarray(forecast["joint_pmf"], dtype=float)
        final_grid = update_pregame_grid_to_live(pregame_grid, live_state, live_cfg)
        snapshots.append(
            {
                "match_id": match_id,
                "live_state": live_state,
                "joint_pmf": final_grid.tolist(),
            }
        )
    return snapshots
