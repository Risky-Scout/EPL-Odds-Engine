from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
import math

import numpy as np


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    total = float(np.sum(grid))
    if total <= 0:
        raise ValueError("grid mass must be positive")
    return grid / total


def clip_grid(grid: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(grid, eps, None)
    return normalize_grid(clipped)


def goal_expectations(grid: np.ndarray) -> Tuple[float, float]:
    x = np.arange(grid.shape[0], dtype=float)
    y = np.arange(grid.shape[1], dtype=float)
    home = float(np.sum(grid * x[:, None]))
    away = float(np.sum(grid * y[None, :]))
    return home, away


def one_x_two_from_grid(grid: np.ndarray) -> Dict[str, float]:
    home = draw = away = 0.0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            p = float(grid[x, y])
            if x > y:
                home += p
            elif x == y:
                draw += p
            else:
                away += p
    return {"home": home, "draw": draw, "away": away}


def totals_prob_from_grid(grid: np.ndarray, line: float, side: str = "over") -> float:
    total = 0.0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            goals = x + y
            if side == "over" and goals > line:
                total += float(grid[x, y])
            elif side == "under" and goals < line or (side == "under" and line.is_integer() and goals <= int(line)):
                total += float(grid[x, y])
    if side == "under" and float(line).is_integer():
        # under 2.0 means <= 2, but under 2.5 means < 2.5
        pass
    return total


def totals_market(grid: np.ndarray, line: float) -> Dict[str, float]:
    over = 0.0
    under = 0.0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            goals = x + y
            p = float(grid[x, y])
            if goals > line:
                over += p
            else:
                under += p
    return {"over": over, "under": under}


def btts_from_grid(grid: np.ndarray) -> Dict[str, float]:
    yes = float(np.sum(grid[1:, 1:]))
    return {"yes": yes, "no": 1.0 - yes}


def correct_score_topn(grid: np.ndarray, n: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            out.append({"home_goals": x, "away_goals": y, "probability": float(grid[x, y])})
    out.sort(key=lambda item: item["probability"], reverse=True)
    return out[:n]


def grid_log_loss(grid: np.ndarray, home_goals: int, away_goals: int, eps: float = 1e-12) -> float:
    x = min(home_goals, grid.shape[0] - 1)
    y = min(away_goals, grid.shape[1] - 1)
    p = float(max(grid[x, y], eps))
    return -math.log(p)


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def decimal_to_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be > 1")
    return 1.0 / decimal_odds


def three_way_implied_probs_decimal(home: float, draw: float, away: float, method: str = "proportional") -> Dict[str, float]:
    raw = np.array(
        [decimal_to_probability(home), decimal_to_probability(draw), decimal_to_probability(away)],
        dtype=float,
    )
    if method != "proportional":
        raise ValueError(f"Unsupported no-vig method: {method}")
    fair = raw / raw.sum()
    return {"home": float(fair[0]), "draw": float(fair[1]), "away": float(fair[2])}


def model_edge(decimal_odds: float, model_probability: float) -> float:
    return model_probability * decimal_odds - 1.0


def fractional_kelly(decimal_odds: float, model_probability: float, fraction: float = 0.2) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    full = (b * model_probability - q) / b if b > 0 else 0.0
    return max(0.0, full) * fraction


def select_value_bets(
    grid: np.ndarray,
    odds_row: Dict[str, float],
    min_edge: float,
    min_decimal_odds: float,
    max_decimal_odds: float,
    kelly_fraction: float,
) -> List[Dict[str, float]]:
    one_x_two = one_x_two_from_grid(grid)
    bets: List[Dict[str, float]] = []
    for side, model_prob in one_x_two.items():
        market_odds = float(odds_row.get(f"odds_{side}", np.nan))
        if not np.isfinite(market_odds):
            continue
        if not (min_decimal_odds <= market_odds <= max_decimal_odds):
            continue
        edge = model_edge(market_odds, model_prob)
        if edge >= min_edge:
            bets.append(
                {
                    "market": "1x2",
                    "selection": side,
                    "decimal_odds": market_odds,
                    "model_probability": model_prob,
                    "edge": edge,
                    "stake_fraction": fractional_kelly(market_odds, model_prob, kelly_fraction),
                }
            )
    bets.sort(key=lambda item: item["edge"], reverse=True)
    return bets


def derived_markets(grid: np.ndarray) -> Dict[str, Any]:
    home_exp, away_exp = goal_expectations(grid)
    return {
        "1x2": one_x_two_from_grid(grid),
        "totals_2_5": totals_market(grid, 2.5),
        "btts": btts_from_grid(grid),
        "top_correct_scores": correct_score_topn(grid, 10),
        "goal_expectations": {"home": home_exp, "away": away_exp},
    }
