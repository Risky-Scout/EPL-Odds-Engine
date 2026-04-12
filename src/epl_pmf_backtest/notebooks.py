from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from .config import ProjectConfig


def build_research_notebook(config: ProjectConfig) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# EPL Joint PMF Walk-Forward Analysis\n\n"
            "This notebook inspects the latest backtest artifact, plots the main diagnostics, "
            "and prepares publication-ready outputs for GitHub and WizardOfOdds."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import ast\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from epl_pmf_backtest.config import ProjectConfig\n"
            "from epl_pmf_backtest.reporting import run_report\n"
            f"config = ProjectConfig.from_yaml('{config.storage_root.as_posix()}/../configs/epl_walkforward.example.yaml')\n"
        ),
        nbf.v4.new_code_cell(
            "root = config.storage_root\n"
            "latest = pd.read_json(root / 'backtests' / 'latest_run.json', typ='series')\n"
            "predictions = pd.read_csv(latest['predictions_path'])\n"
            "predictions.head()"
        ),
        nbf.v4.new_code_cell(
            "if 'joint_pmf_json' in predictions.columns:\n"
            "    predictions['joint_pmf_json'] = predictions['joint_pmf_json'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)\n"
            "predictions[['date', 'home_team', 'away_team', 'joint_log_loss']].tail()"
        ),
        nbf.v4.new_code_cell("run_report(config)"),
    ]
    out = config.storage_root / "reports" / "01_historical_walkforward_analysis.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out
