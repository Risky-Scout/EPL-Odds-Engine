
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PromotedPriorContext:
    seen_teams: set[str]
    epl_baselines: Dict[str, float]
    epl_strengths: Dict[str, Dict[str, float]]
    lower_baselines: Dict[str, float]
    transition_multipliers: Dict[str, float]
    lower_division_by_team_season: Dict[tuple[str, str], Dict[str, float]]
    lower_division_league: str
    top_league: str
    prior_matches_for_transition: int
    strength_shrink_matches: float


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    lam = max(float(lam), 1e-9)
    return np.exp(-lam) * np.power(lam, k) / np.vectorize(math.factorial)(k)


def previous_season(season: str) -> str:
    season = str(season)
    parts = season.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unsupported season format: {season}")
    return f"{int(parts[0]) - 1}-{int(parts[1]) - 1}"


def _league_baselines(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "home_goals_for": float(df["goals_home"].mean()),
        "away_goals_for": float(df["goals_away"].mean()),
        "home_goals_against": float(df["goals_away"].mean()),
        "away_goals_against": float(df["goals_home"].mean()),
    }


def _team_strength_table(df: pd.DataFrame, shrink_matches: float) -> Dict[str, Dict[str, float]]:
    base = _league_baselines(df)
    out: Dict[str, Dict[str, float]] = {}

    home = (
        df.groupby("team_home")
        .agg(home_matches=("team_home", "size"),
             home_gf=("goals_home", "sum"),
             home_ga=("goals_away", "sum"))
        .reset_index()
        .rename(columns={"team_home": "team"})
    )
    away = (
        df.groupby("team_away")
        .agg(away_matches=("team_away", "size"),
             away_gf=("goals_away", "sum"),
             away_ga=("goals_home", "sum"))
        .reset_index()
        .rename(columns={"team_away": "team"})
    )
    merged = pd.merge(home, away, on="team", how="outer").fillna(0.0)

    for row in merged.itertuples(index=False):
        hm = float(row.home_matches)
        am = float(row.away_matches)
        out[str(row.team)] = {
            "home_attack": ((float(row.home_gf) + shrink_matches * base["home_goals_for"]) / max(hm + shrink_matches, 1e-9)) / max(base["home_goals_for"], 1e-9),
            "home_defense": ((float(row.home_ga) + shrink_matches * base["home_goals_against"]) / max(hm + shrink_matches, 1e-9)) / max(base["home_goals_against"], 1e-9),
            "away_attack": ((float(row.away_gf) + shrink_matches * base["away_goals_for"]) / max(am + shrink_matches, 1e-9)) / max(base["away_goals_for"], 1e-9),
            "away_defense": ((float(row.away_ga) + shrink_matches * base["away_goals_against"]) / max(am + shrink_matches, 1e-9)) / max(base["away_goals_against"], 1e-9),
            "home_matches": hm,
            "away_matches": am,
        }
    return out


def _build_lower_division_team_season_lookup(lower_division_df: pd.DataFrame, shrink_matches: float) -> tuple[Dict[tuple[str, str], Dict[str, float]], Dict[str, float]]:
    lower_division_df = lower_division_df.copy()
    out: Dict[tuple[str, str], Dict[str, float]] = {}
    baselines = _league_baselines(lower_division_df)
    for season, season_df in lower_division_df.groupby("season"):
        strengths = _team_strength_table(season_df, shrink_matches=shrink_matches)
        for team, vals in strengths.items():
            out[(str(season), str(team))] = vals
    return out, baselines


def _promoted_teams_in_train(top_df: pd.DataFrame) -> Dict[str, str]:
    seasons = sorted({str(x) for x in top_df["season"].dropna().unique()})
    by_season = {season: set() for season in seasons}
    for season, season_df in top_df.groupby("season"):
        teams = set(season_df["team_home"]).union(set(season_df["team_away"]))
        by_season[str(season)] = teams
    promoted: Dict[str, str] = {}
    for season in seasons:
        prev = previous_season(season)
        prev_teams = by_season.get(prev, set())
        curr_teams = by_season.get(season, set())
        for team in sorted(curr_teams - prev_teams):
            promoted[f"{season}::{team}"] = team
    return promoted


def _learn_transition_multipliers(
    top_df: pd.DataFrame,
    lower_division_lookup: Dict[tuple[str, str], Dict[str, float]],
    shrink_matches: float,
    prior_matches_for_transition: int,
) -> Dict[str, float]:
    multipliers = {
        "home_attack": [],
        "home_defense": [],
        "away_attack": [],
        "away_defense": [],
    }

    promoted = _promoted_teams_in_train(top_df)
    for season_team_key, team in promoted.items():
        season, _ = season_team_key.split("::", 1)
        prev = previous_season(season)
        lower = lower_division_lookup.get((prev, team))
        if lower is None:
            continue

        season_top_df = top_df[top_df["season"].astype(str) == season].copy()
        team_rows = season_top_df[(season_top_df["team_home"] == team) | (season_top_df["team_away"] == team)].sort_values("date")
        team_rows = team_rows.head(prior_matches_for_transition)
        if len(team_rows) < max(4, prior_matches_for_transition // 2):
            continue

        epl_strengths = _team_strength_table(team_rows, shrink_matches=shrink_matches)
        top = epl_strengths.get(team)
        if top is None:
            continue

        for key in multipliers:
            base = float(lower.get(key, 1.0))
            if base <= 0:
                continue
            multipliers[key].append(float(top.get(key, 1.0)) / base)

    out: Dict[str, float] = {}
    for key, vals in multipliers.items():
        if vals:
            out[key] = float(np.median(np.asarray(vals, dtype=float)))
        else:
            out[key] = 1.0
    return out


def build_promoted_prior_context(
    top_league_train_df: pd.DataFrame,
    lower_division_df: pd.DataFrame | None,
    *,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
    prior_matches_for_transition: int = 10,
    strength_shrink_matches: float = 8.0,
) -> PromotedPriorContext | None:
    if lower_division_df is None or len(lower_division_df) == 0:
        return None

    top_league_train_df = top_league_train_df.copy().sort_values("date")
    lower_division_df = lower_division_df.copy().sort_values("date")

    epl_baselines = _league_baselines(top_league_train_df)
    epl_strengths = _team_strength_table(top_league_train_df, shrink_matches=strength_shrink_matches)
    lower_lookup, lower_baselines = _build_lower_division_team_season_lookup(
        lower_division_df,
        shrink_matches=strength_shrink_matches,
    )
    transition = _learn_transition_multipliers(
        top_league_train_df,
        lower_lookup,
        shrink_matches=strength_shrink_matches,
        prior_matches_for_transition=prior_matches_for_transition,
    )

    seen_teams = set(top_league_train_df["team_home"]).union(set(top_league_train_df["team_away"]))
    return PromotedPriorContext(
        seen_teams=seen_teams,
        epl_baselines=epl_baselines,
        epl_strengths=epl_strengths,
        lower_baselines=lower_baselines,
        transition_multipliers=transition,
        lower_division_by_team_season=lower_lookup,
        lower_division_league=lower_division_league,
        top_league=top_league,
        prior_matches_for_transition=int(prior_matches_for_transition),
        strength_shrink_matches=float(strength_shrink_matches),
    )


def _translated_strengths_for_team(team: str, target_top_season: str, ctx: PromotedPriorContext) -> Dict[str, float] | None:
    prior_season = previous_season(target_top_season)
    lower = ctx.lower_division_by_team_season.get((prior_season, team))
    if lower is None:
        return None

    out: Dict[str, float] = {}
    for key in ("home_attack", "home_defense", "away_attack", "away_defense"):
        raw = float(lower.get(key, 1.0))
        translated = raw * float(ctx.transition_multipliers.get(key, 1.0))
        # Conservative shrinkage toward league average for cross-division translation.
        out[key] = 0.35 * 1.0 + 0.65 * translated
    return out


def build_real_data_prior_grid(
    train_df: pd.DataFrame,
    lower_division_df: pd.DataFrame | None,
    *,
    home_team: str,
    away_team: str,
    match_season: str,
    max_goals: int,
    top_league: str = "ENG Premier League",
    lower_division_league: str = "ENG Championship",
    context: PromotedPriorContext | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    ctx = context or build_promoted_prior_context(
        train_df,
        lower_division_df,
        top_league=top_league,
        lower_division_league=lower_division_league,
    )
    if ctx is None:
        raise RuntimeError(
            f"UNSEEN_TEAM_IN_FOLD: {home_team} vs {away_team}. "
            "No lower-division data were available to build a real promoted-team prior."
        )

    home_strength = ctx.epl_strengths.get(home_team)
    away_strength = ctx.epl_strengths.get(away_team)

    if home_strength is None:
        home_strength = _translated_strengths_for_team(home_team, str(match_season), ctx)
    if away_strength is None:
        away_strength = _translated_strengths_for_team(away_team, str(match_season), ctx)

    if home_strength is None or away_strength is None:
        missing = []
        if home_strength is None:
            missing.append(home_team)
        if away_strength is None:
            missing.append(away_team)
        raise RuntimeError(
            f"UNSEEN_TEAM_IN_FOLD: {home_team} vs {away_team}. "
            f"Missing real lower-division priors for: {', '.join(missing)}."
        )

    lam_home = ctx.epl_baselines["home_goals_for"] * home_strength["home_attack"] * away_strength["away_defense"]
    lam_away = ctx.epl_baselines["away_goals_for"] * away_strength["away_attack"] * home_strength["home_defense"]

    lam_home = max(0.05, min(float(lam_home), 4.5))
    lam_away = max(0.05, min(float(lam_away), 4.5))

    k = np.arange(max_goals + 1)
    home_pmf = _poisson_pmf(k, lam_home)
    away_pmf = _poisson_pmf(k, lam_away)
    grid = np.outer(home_pmf, away_pmf)
    grid = grid / grid.sum()

    meta = {
        "source": "real_data_promoted_team_prior",
        "home_goal_expectation": lam_home,
        "away_goal_expectation": lam_away,
        "top_league": ctx.top_league,
        "lower_division_league": ctx.lower_division_league,
        "transition_multipliers": dict(ctx.transition_multipliers),
        "home_team_seen_in_top_league_train": home_team in ctx.seen_teams,
        "away_team_seen_in_top_league_train": away_team in ctx.seen_teams,
    }
    return grid, meta
