from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import re

import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning, module=r"penaltyblog\\.scrapers\\..*")

from ..storage import write_csv


_HOME_CANDIDATES = [
    "{prefix}_h", "{prefix}h", "{prefix}_home", "{prefix}home",
]
_DRAW_CANDIDATES = [
    "{prefix}_d", "{prefix}d", "{prefix}_draw", "{prefix}draw",
]
_AWAY_CANDIDATES = [
    "{prefix}_a", "{prefix}a", "{prefix}_away", "{prefix}away",
]


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        col = lower_map.get(candidate.lower())
        if col is not None:
            return str(col)
    return None


def _select_three_way_odds(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out["odds_home"] = pd.NA
    out["odds_draw"] = pd.NA
    out["odds_away"] = pd.NA

    for prefix in prefixes:
        home_col = _first_existing(df, [c.format(prefix=prefix) for c in _HOME_CANDIDATES])
        draw_col = _first_existing(df, [c.format(prefix=prefix) for c in _DRAW_CANDIDATES])
        away_col = _first_existing(df, [c.format(prefix=prefix) for c in _AWAY_CANDIDATES])
        if home_col and draw_col and away_col:
            out["odds_home"] = out["odds_home"].fillna(pd.to_numeric(df[home_col], errors="coerce"))
            out["odds_draw"] = out["odds_draw"].fillna(pd.to_numeric(df[draw_col], errors="coerce"))
            out["odds_away"] = out["odds_away"].fillna(pd.to_numeric(df[away_col], errors="coerce"))
    return out


def _slugify_league(league: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(league).lower()).strip("_")
    return slug or "league"


class HistoricalFootballDataProvider:
    """Historical EPL data provider using Penaltyblog's football-data.co.uk scraper."""

    def __init__(self, league: str, seasons: Iterable[str], odds_priority: Iterable[str], storage_root: Path):
        self.league = league
        self.seasons = list(seasons)
        self.odds_priority = list(odds_priority)
        self.storage_root = storage_root

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        curated_path = self.storage_root / "curated" / f"historical_matches__{_slugify_league(self.league)}.csv"
        if use_cache and curated_path.exists():
            df = pd.read_csv(curated_path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        try:
            import penaltyblog as pb
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("penaltyblog is required to fetch historical data") from exc

        frames = []
        for season in self.seasons:
            scraper = pb.scrapers.FootballData(self.league, season)
            raw = scraper.get_fixtures().copy()
            df = raw.copy()
            lower = {c: str(c).lower() for c in df.columns}
            df = df.rename(columns=lower)

            if "goals_home" not in df.columns:
                if "fthg" in df.columns:
                    df["goals_home"] = pd.to_numeric(df["fthg"], errors="coerce")
            if "goals_away" not in df.columns:
                if "ftag" in df.columns:
                    df["goals_away"] = pd.to_numeric(df["ftag"], errors="coerce")

            required = ["date", "team_home", "team_away", "goals_home", "goals_away"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns from historical feed for season {season}: {missing}")

            df = _select_three_way_odds(df, self.odds_priority)
            df["season"] = df.get("season", season)
            df["competition"] = df.get("competition", self.league)
            df["match_id"] = (
                df.index.astype(str)
                if "id" not in df.columns
                else df["id"].astype(str)
            )
            keep = [
                "match_id",
                "date",
                "season",
                "competition",
                "team_home",
                "team_away",
                "goals_home",
                "goals_away",
                "odds_home",
                "odds_draw",
                "odds_away",
            ]
            frames.append(df[keep].copy())

        out = pd.concat(frames, ignore_index=True).copy()
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out["goals_home"] = pd.to_numeric(out["goals_home"], errors="coerce")
        out["goals_away"] = pd.to_numeric(out["goals_away"], errors="coerce")
        out["odds_home"] = pd.to_numeric(out["odds_home"], errors="coerce")
        out["odds_draw"] = pd.to_numeric(out["odds_draw"], errors="coerce")
        out["odds_away"] = pd.to_numeric(out["odds_away"], errors="coerce")
        out = out.dropna(subset=["date", "team_home", "team_away", "goals_home", "goals_away"])
        out = out.sort_values("date").reset_index(drop=True)

        write_csv(curated_path, out)
        return out
