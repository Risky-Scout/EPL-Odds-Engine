from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


DIRS = [
    "raw",
    "curated",
    "features",
    "backtests",
    "reports",
    "figures",
    "live",
    "exports",
    "sample_outputs",
    "cache/http",
]


def ensure_dirs(root: Path) -> None:
    for rel in DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def latest_matching(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None
