from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


@dataclass(slots=True)
class ProjectConfig:
    raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=_expand(raw))

    @property
    def storage_root(self) -> Path:
        configured = self.raw.get("storage", {}).get("root_dir")
        env_override = os.getenv("EPL_PMF_STORAGE_ROOT")
        root = env_override or configured or "./workspace/epl_joint_pmf"
        return Path(root)

    @property
    def league(self) -> str:
        return str(self.raw.get("project", {}).get("league", "ENG Premier League"))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("project", {}).get("random_seed", 42))

    @property
    def model_version(self) -> str:
        return str(self.raw.get("project", {}).get("model_version", "0.0.0"))

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get_api_key(self) -> str | None:
        live = self.raw.get("providers", {}).get("live", {})
        env_name = live.get("api_key_env")
        return os.getenv(env_name) if env_name else None

    def as_dict(self) -> Dict[str, Any]:
        return self.raw
