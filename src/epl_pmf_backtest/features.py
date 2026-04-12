from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable
import math

import numpy as np
import pandas as pd


def add_pi_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import penaltyblog as pb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("penaltyblog is required for Pi ratings features") from exc

    out = df.sort_values("date").copy()
    pi = pb.ratings.PiRatingSystem()

    pre_home = []
    pre_away = []
    pi_home_win = []
    pi_draw = []
    pi_away_win = []

    for row in out.itertuples(index=False):
        home = str(row.team_home)
        away = str(row.team_away)

        home_rating = float(pi.get_team_rating(home))
        away_rating = float(pi.get_team_rating(away))
        probs = pi.calculate_match_probabilities(home, away)

        pre_home.append(home_rating)
        pre_away.append(away_rating)
        pi_home_win.append(float(probs["home_win"]))
        pi_draw.append(float(probs["draw"]))
        pi_away_win.append(float(probs["away_win"]))

        goal_diff = int(row.goals_home) - int(row.goals_away)
        pi.update_ratings(home, away, goal_diff)

    out["pi_home_rating"] = pre_home
    out["pi_away_rating"] = pre_away
    out["pi_rating_diff"] = out["pi_home_rating"] - out["pi_away_rating"]
    out["pi_home_win"] = pi_home_win
    out["pi_draw"] = pi_draw
    out["pi_away_win"] = pi_away_win
    return out


def add_rolling_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.sort_values("date").copy()

    goals_for: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    goals_against: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    points: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    home_for = []
    home_against = []
    away_for = []
    away_against = []
    home_points = []
    away_points = []

    for row in out.itertuples(index=False):
        home = str(row.team_home)
        away = str(row.team_away)

        home_for.append(float(np.mean(goals_for[home])) if goals_for[home] else np.nan)
        home_against.append(float(np.mean(goals_against[home])) if goals_against[home] else np.nan)
        away_for.append(float(np.mean(goals_for[away])) if goals_for[away] else np.nan)
        away_against.append(float(np.mean(goals_against[away])) if goals_against[away] else np.nan)
        home_points.append(float(np.mean(points[home])) if points[home] else np.nan)
        away_points.append(float(np.mean(points[away])) if points[away] else np.nan)

        gh = int(row.goals_home)
        ga = int(row.goals_away)
        goals_for[home].append(gh)
        goals_against[home].append(ga)
        goals_for[away].append(ga)
        goals_against[away].append(gh)

        if gh > ga:
            hp, ap = 3, 0
        elif gh == ga:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3
        points[home].append(hp)
        points[away].append(ap)

    out["rolling_home_goals_for"] = home_for
    out["rolling_home_goals_against"] = home_against
    out["rolling_away_goals_for"] = away_for
    out["rolling_away_goals_against"] = away_against
    out["rolling_home_points"] = home_points
    out["rolling_away_points"] = away_points
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_pi_features(df)
    out = add_rolling_form_features(out, window=5)
    fill_cols = [
        "rolling_home_goals_for",
        "rolling_home_goals_against",
        "rolling_away_goals_for",
        "rolling_away_goals_against",
        "rolling_home_points",
        "rolling_away_points",
    ]
    out[fill_cols] = out[fill_cols].fillna(out[fill_cols].mean())
    return out
