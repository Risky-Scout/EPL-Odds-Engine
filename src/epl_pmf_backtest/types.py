from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class ForecastGrid:
    match_id: str
    home_team: str
    away_team: str
    kickoff_utc: str
    max_goals: int
    joint_pmf: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def actual_mass(self) -> float:
        return float(sum(sum(row) for row in self.joint_pmf))


@dataclass(slots=True)
class FoldWindow:
    fold_id: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass(slots=True)
class BacktestSummary:
    run_id: str
    folds: List[Dict[str, Any]]
    overall: Dict[str, Any]
    artifacts: Dict[str, str]
