from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable
import hashlib
import json
import time
import urllib.parse

import requests


class BDLRateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.time()
        while self.calls and now - self.calls[0] > 60.0:
            self.calls.popleft()
        if len(self.calls) >= self.per_minute:
            sleep_for = 60.0 - (now - self.calls[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.calls.append(time.time())


class BDLEPLClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        cache_dir: Path,
        per_page: int = 100,
        rate_limit_per_minute: int = 480,
        cache_ttl_seconds: int = 30,
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.per_page = per_page
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.rate_limiter = BDLRateLimiter(rate_limit_per_minute)

    def _cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        raw = f"{endpoint}?{urllib.parse.urlencode(sorted(params.items()), doseq=True)}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _get(self, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = dict(params or {})
        params.setdefault("per_page", self.per_page)
        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= self.cache_ttl_seconds:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self.rate_limiter.wait()
        response = requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            params=params,
            headers={"Authorization": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def get_matches(self, dates: Iterable[str] | None = None, match_ids: Iterable[int] | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if dates:
            params["dates[]"] = list(dates)
        if match_ids:
            params["match_ids[]"] = list(match_ids)
        return self._get("matches", params)

    def get_odds(self, dates: Iterable[str] | None = None, match_ids: Iterable[int] | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if dates:
            params["dates[]"] = list(dates)
        if match_ids:
            params["match_ids[]"] = list(match_ids)
        return self._get("odds", params)

    def get_match_events(self, match_ids: Iterable[int]) -> Dict[str, Any]:
        return self._get("match_events", {"match_ids[]": list(match_ids)})

    def get_match_lineups(self, match_ids: Iterable[int]) -> Dict[str, Any]:
        return self._get("match_lineups", {"match_ids[]": list(match_ids)})

    def get_team_match_stats(self, match_ids: Iterable[int]) -> Dict[str, Any]:
        return self._get("team_match_stats", {"match_ids[]": list(match_ids)})

    def build_live_state(self, match_id: int) -> Dict[str, Any]:
        matches = self.get_matches(match_ids=[match_id]).get("data", [])
        odds = self.get_odds(match_ids=[match_id]).get("data", [])
        events = self.get_match_events([match_id]).get("data", [])
        lineups = self.get_match_lineups([match_id]).get("data", [])
        team_stats = self.get_team_match_stats([match_id]).get("data", [])

        return {
            "match": matches[0] if matches else {},
            "odds": odds,
            "events": events,
            "lineups": lineups,
            "team_match_stats": team_stats,
        }
