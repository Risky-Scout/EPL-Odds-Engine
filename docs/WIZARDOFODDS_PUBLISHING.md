# WizardOfOdds Publishing

The publishing format centers on a single object per match:

- match metadata
- model version
- timestamp
- joint PMF
- derived markets
- market comparison
- edge summary
- notes on calibration and data freshness

The exporter writes a compact JSON file that can be pushed directly to your website workflow.

Recommended publishing fields:

- `match_id`
- `competition`
- `home_team`
- `away_team`
- `kickoff_utc`
- `model_version`
- `joint_pmf`
- `derived_markets`
- `market_reference`
- `edge_summary`
- `live_state`
- `generated_at_utc`
