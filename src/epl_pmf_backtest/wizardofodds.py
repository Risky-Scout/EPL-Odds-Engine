from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np

from .markets import derived_markets


def forecast_to_wizardofodds_payload(forecast: Dict[str, Any], market_reference: Dict[str, Any] | None = None) -> Dict[str, Any]:
    grid = np.asarray(forecast["joint_pmf"], dtype=float)
    return {
        "match_id": forecast["match_id"],
        "home_team": forecast["home_team"],
        "away_team": forecast["away_team"],
        "kickoff_utc": forecast["kickoff_utc"],
        "model_version": forecast.get("model_version", "unknown"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "joint_pmf": forecast["joint_pmf"],
        "derived_markets": derived_markets(grid),
        "market_reference": market_reference or {},
        "metadata": {
            "calibration_temperature": forecast.get("calibration_temperature"),
            "pi_blend_weight": forecast.get("pi_blend_weight"),
            "ensemble_weights": forecast.get("ensemble_weights"),
        },
    }
