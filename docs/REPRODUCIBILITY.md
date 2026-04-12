# Reproducibility

Every run should be tied to:

- a config file
- code version / Git commit
- storage root
- data snapshot date
- random seed
- model set
- calibration method
- odds de-vig method
- bet filter thresholds

Recommended workflow:

1. Sync historical data.
2. Freeze curated input tables for the run.
3. Run walk-forward backtest.
4. Generate report and notebook.
5. Publish summary figures and selected JSON outputs.
6. Keep raw logs and per-match predictions for auditability.

The report generator writes the run configuration into the summary JSON so that WizardOfOdds-facing outputs can always be traced back to the source run.
