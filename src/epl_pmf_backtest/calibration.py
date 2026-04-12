from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .markets import clip_grid, grid_log_loss


def temperature_scale_grid(grid: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.power(np.clip(grid, 1e-12, None), 1.0 / temperature)
    return clip_grid(scaled)


def fit_grid_temperature(
    grids: Sequence[np.ndarray],
    outcomes: Sequence[Tuple[int, int]],
    candidates: Iterable[float],
) -> tuple[float, float]:
    best_t = 1.0
    best_loss = float("inf")
    for t in candidates:
        losses = []
        for grid, (gh, ga) in zip(grids, outcomes):
            calibrated = temperature_scale_grid(grid, float(t))
            losses.append(grid_log_loss(calibrated, int(gh), int(ga)))
        avg = float(np.mean(losses)) if losses else float("inf")
        if avg < best_loss:
            best_loss = avg
            best_t = float(t)
    return best_t, best_loss


def reliability_curve(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = []
    observed = []
    counts = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (probabilities >= lo) & (probabilities < hi if i < bins - 1 else probabilities <= hi)
        if mask.sum() == 0:
            centers.append((lo + hi) / 2.0)
            observed.append(np.nan)
            counts.append(0)
        else:
            centers.append(float(probabilities[mask].mean()))
            observed.append(float(outcomes[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(observed), np.asarray(counts)
