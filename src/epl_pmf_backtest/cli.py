from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .backtest import run_backtest, sync_historical
from .config import ProjectConfig
from .live import build_live_snapshot
from .providers.live_bdl import BDLEPLClient
from .reporting import run_report
from .sample import build_sample_run
from .storage import ensure_dirs, read_json, write_json
from .notebooks import build_research_notebook


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="epl-pmf")
    sub = p.add_subparsers(dest="command", required=True)

    command_names = (
        "sync-historical",
        "sync",
        "backtest",
        "report",
        "live-snapshot",
        "live",
        "build-sample-run",
        "sample",
        "build-notebook",
        "notebook",
    )
    for name in command_names:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
    return p


def main() -> None:
    args = _parser().parse_args()
    config = ProjectConfig.from_yaml(args.config)
    ensure_dirs(config.storage_root)

    if args.command in {"sync-historical", "sync"}:
        out = sync_historical(config)
        print(f"Historical sync written to: {out}")
        return

    if args.command == "backtest":
        out = run_backtest(config)
        print(f"Backtest summary written to: {out}")
        return

    if args.command == "report":
        out = run_report(config)
        print(f"Report written to: {out}")
        return

    if args.command in {"build-sample-run", "sample"}:
        out = build_sample_run(config)
        print(f"Sample run written to: {out}")
        return

    if args.command in {"build-notebook", "notebook"}:
        out = build_research_notebook(config)
        print(f"Notebook written to: {out}")
        return

    if args.command in {"live-snapshot", "live"}:
        latest = read_json(config.storage_root / "backtests" / "latest_run.json")
        predictions_path = Path(latest["predictions_path"])
        import pandas as pd

        df = pd.read_csv(predictions_path)
        import ast
        df["joint_pmf_json"] = df["joint_pmf_json"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        api_key = config.get_api_key()
        if not api_key:
            raise RuntimeError("Missing BDL API key. Set the configured environment variable before running live-snapshot.")

        live_cfg = config.section("live")
        provider_cfg = config.raw.get("providers", {}).get("live", {})
        client = BDLEPLClient(
            api_key=api_key,
            base_url=str(provider_cfg.get("base_url")),
            cache_dir=config.storage_root / "cache" / "http",
            per_page=int(provider_cfg.get("per_page", 100)),
            rate_limit_per_minute=int(provider_cfg.get("rate_limit_per_minute", 480)),
            cache_ttl_seconds=int(provider_cfg.get("cache_ttl_seconds", 15)),
        )

        latest_forecasts = []
        for row in df.tail(10).itertuples(index=False):
            latest_forecasts.append(
                {
                    "match_id": row.match_id,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "kickoff_utc": row.date,
                    "joint_pmf": row.joint_pmf_json,
                }
            )

        snapshots = build_live_snapshot(client, latest_forecasts, live_cfg)
        out = config.storage_root / "live" / "latest_live_snapshot.json"
        write_json(out, snapshots)
        print(f"Live snapshot written to: {out}")
        return


if __name__ == "__main__":
    main()
