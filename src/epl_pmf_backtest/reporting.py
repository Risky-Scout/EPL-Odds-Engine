from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import reliability_curve
from .config import ProjectConfig
from .storage import ensure_dirs, latest_matching, read_json, write_json


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #111; }
    h1, h2 { margin-bottom: 0.3rem; }
    .meta { color: #555; margin-bottom: 1.5rem; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 20px; }
    .card { border: 1px solid #ddd; border-radius: 14px; padding: 14px; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #eee; }
    code { background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px 8px; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="meta">Generated {{ generated_at }}</div>

  <div class="card">
    <h2>Overall summary</h2>
    <table>
      {% for key, value in summary.items() %}
      <tr><th>{{ key }}</th><td>{{ value }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <h2>Figures</h2>
  <div class="grid">
    {% for figure in figures %}
    <div class="card">
      <h3>{{ figure.title }}</h3>
      <img src="{{ figure.path }}" alt="{{ figure.title }}" />
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""


def _load_latest_run(root: Path) -> Tuple[Path, Path]:
    latest_path = root / "backtests" / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No completed backtest run was found at {latest_path}. "
            "Run `epl-pmf backtest --config <config>` successfully before `epl-pmf report`."
        )
    latest = read_json(latest_path)
    return Path(latest["predictions_path"]), Path(latest["summary_path"])


def _plot_rolling_log_loss(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    rolling = df["joint_log_loss"].rolling(50, min_periods=5).mean()
    plt.plot(pd.to_datetime(df["date"]), rolling, linewidth=2)
    plt.title("Rolling Joint Log Loss (50-match mean)")
    plt.xlabel("Date")
    plt.ylabel("Joint log loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_calibration(df: pd.DataFrame, out_path: Path) -> None:
    probs = df["model_home_win"].to_numpy(dtype=float)
    outcomes = df["actual_home_win"].to_numpy(dtype=float)
    x, y, counts = reliability_curve(probs, outcomes, bins=10)

    plt.figure(figsize=(6.5, 6.5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.scatter(x, y, s=np.maximum(counts, 1) * 8, alpha=0.9)
    plt.title("Calibration: Home-win probability")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_model_vs_market(df: pd.DataFrame, out_path: Path) -> None:
    market = df["market_fair_home"].to_numpy(dtype=float)
    model = df["model_home_win"].to_numpy(dtype=float)
    mask = np.isfinite(market) & np.isfinite(model)
    plt.figure(figsize=(7, 7))
    plt.scatter(market[mask], model[mask], alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.title("Model vs Market: Home-win fair probabilities")
    plt.xlabel("Market fair probability")
    plt.ylabel("Model probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_equity_curve(df: pd.DataFrame, out_path: Path) -> None:
    profit = df["bet_result_profit_per_unit"].fillna(0.0) * df["bet_stake_fraction"].fillna(0.0)
    equity = profit.cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(df["date"]), equity, linewidth=2)
    plt.title("Equity Curve (fractional Kelly, research backtest)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit per unit bankroll")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_joint_pmf_example(df: pd.DataFrame, out_path: Path) -> None:
    row = df.sort_values("bet_edge", ascending=False, na_position="last").iloc[0]
    grid = np.asarray(row["joint_pmf_json"], dtype=float)
    plt.figure(figsize=(7, 6))
    plt.imshow(grid, origin="lower", aspect="auto")
    plt.title(f"Joint PMF Example: {row['home_team']} vs {row['away_team']}")
    plt.xlabel("Away goals")
    plt.ylabel("Home goals")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_report(config: ProjectConfig) -> Path:
    root = config.storage_root
    ensure_dirs(root)

    predictions_path, summary_path = _load_latest_run(root)
    df = pd.read_csv(predictions_path)
    summary = read_json(summary_path)

    def _parse_jsonish(series: pd.Series) -> pd.Series:
        import ast
        return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    if "joint_pmf_json" in df.columns:
        df["joint_pmf_json"] = _parse_jsonish(df["joint_pmf_json"])

    figures_dir = root / "figures"
    rolling_path = figures_dir / "rolling_joint_log_loss.png"
    calibration_path = figures_dir / "calibration_home_win.png"
    model_market_path = figures_dir / "model_vs_market_home.png"
    equity_path = figures_dir / "equity_curve.png"
    pmf_path = figures_dir / "joint_pmf_example.png"

    _plot_rolling_log_loss(df, rolling_path)
    _plot_calibration(df, calibration_path)
    _plot_model_vs_market(df, model_market_path)
    _plot_equity_curve(df, equity_path)
    _plot_joint_pmf_example(df, pmf_path)

    figures = [
        {"title": "Rolling joint log loss", "path": str(rolling_path)},
        {"title": "Calibration", "path": str(calibration_path)},
        {"title": "Model vs market", "path": str(model_market_path)},
        {"title": "Equity curve", "path": str(equity_path)},
        {"title": "Joint PMF example", "path": str(pmf_path)},
    ]

    md_path = root / "reports" / "latest_report.md"
    html_path = root / "reports" / "latest_report.html"

    md_lines = [
        f"# {config.section('reporting').get('title', 'EPL Joint PMF Walk-Forward Report')}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overall summary",
        "",
    ]
    for key, value in summary.items():
        md_lines.append(f"- **{key}**: {value}")
    md_lines.extend([
        "",
        "## Figures",
        "",
        f"![Rolling log loss]({rolling_path})",
        "",
        f"![Calibration]({calibration_path})",
        "",
        f"![Model vs market]({model_market_path})",
        "",
        f"![Equity curve]({equity_path})",
        "",
        f"![Joint PMF example]({pmf_path})",
        "",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    html = jinja2.Template(HTML_TEMPLATE).render(
        title=config.section("reporting").get("title", "EPL Joint PMF Walk-Forward Report"),
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        figures=figures,
    )
    html_path.write_text(html, encoding="utf-8")

    write_json(root / "reports" / "latest_report_manifest.json", {
        "markdown": str(md_path),
        "html": str(html_path),
        "figures": [str(rolling_path), str(calibration_path), str(model_market_path), str(equity_path), str(pmf_path)],
    })
    return html_path
