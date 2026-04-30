# UFC Betting Model

## Project Overview
This repository builds a leakage-aware UFC fight prediction workflow, compares several model families, and runs an exploratory betting backtest against matched market odds.

## Data Sources
- UFC fight-level statistics
- Fighter bio/statistics reference data
- Kaggle UFC betting odds dataset

Large generated datasets and raw odds files are kept out of Git by default for repository cleanliness.
The repo still keeps the full historical combined fight table for EDA, but the predictive workflow uses a narrower audited modeling universe starting in 2010.

## Leakage Prevention
- Pre-fight features only for modeling
- Chronological train / validation / test splits
- Modeling universe restricted to fights on or after `2010-01-01` because earlier rows show a corner-assignment regime break
- Final hold-out evaluation performed once per frozen model family
- Odds ingestion and betting analysis performed after model predictions were frozen

## Feature Engineering
Main engineered groups include:
- cumulative pre-fight stats
- recency and momentum features
- Elo features
- cardio / fight-duration proxies
- style and matchup interaction features
- safe fighter bio attributes

## Models Tested
- Elastic-net logistic regression
- Random Forest
- XGBoost

## Walk-Forward Research
The repo also includes a separate pre-test research workflow that:
- uses the frozen final feature set
- runs annual expanding walk-forward folds on odds-matched pre-test data
- compares raw Random Forest probabilities, calibrated probabilities, market no-vig probabilities, and model-market blends
- keeps the consumed final hold-out benchmark separate from new research

Key outputs:
- `outputs/walk_forward_model_comparison.csv`
- `outputs/walk_forward_market_benchmark_summary.txt`
- `outputs/walk_forward_calibration_comparison.csv`
- `outputs/walk_forward_yearly_metrics.csv`

## Final Model Comparison
The frozen final comparison lives in:
- `outputs/final_model_test_comparison.csv`
- `outputs/final_model_comparison_summary.txt`

At the final hold-out step:
- Logistic regression had the best log loss / Brier score / ROC AUC
- Random Forest had the best accuracy
- XGBoost did not beat either frozen benchmark

## Betting Backtest
The first betting backtest uses:
- frozen Random Forest probabilities
- no-vig implied probabilities from the selected odds source
- flat betting and capped fractional Kelly diagnostics

This backtest is exploratory and not a production betting system. The repo keeps it as a frozen Random Forest betting pass for continuity, even though logistic regression became the strongest final post-cutoff probability model on the consumed hold-out benchmark.

## Why the Strategy Failed
The post-hoc diagnostics suggest the main issues were:
- market efficiency and vig
- imperfect probability calibration for betting use
- weak realized edge quality
- poor underdog / blue-side performance
- reliance on an aggregated odds source (`zewnetrzne`)

See:
- `outputs/betting_failure_diagnostics_summary.txt`
- `outputs/modeling_audit_report.md`

## Limitations
- Some generated datasets are large and excluded from Git
- The raw combined dataset still contains earlier UFC history for EDA, but the modeling pipeline excludes fights before `2010-01-01`
- Excluding pre-2010 fights reduces sample size, but it improves target consistency by avoiding the older corner-assignment regime break
- The betting backtest uses only the odds-matched subset of the hold-out test set
- The selected odds source is aggregated rather than a single directly tradable sharp book
- Betting-threshold rankings on the hold-out set are descriptive only and should not be treated as selected production rules

## Repository Structure
```text
src/
  config.py
  data_utils.py
  features.py
  modeling/
  odds/
  audits/
  train_baseline_models.py
notebooks/
  07_walk_forward_research.ipynb
outputs/
data/
```

`src/train_baseline_models.py` is kept as a compatibility wrapper so existing notebooks and scripts still import the same public entrypoints.

## How to Run
Rebuild feature datasets if needed:
```bash
python3 src/combined_data.py
```

Run the main modeling workflow:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_validation_workflow
run_validation_workflow()
PY
```

Run final frozen model-family comparison:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_final_model_family_comparison
run_final_model_family_comparison()
PY
```

Run odds ingestion:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_odds_ingestion_pipeline
run_odds_ingestion_pipeline()
PY
```

Run betting backtest diagnostics:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_betting_backtest, run_betting_failure_diagnostics
run_betting_backtest()
run_betting_failure_diagnostics()
PY
```

Run the walk-forward calibration and market-benchmark research workflow:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_walk_forward_market_research
run_walk_forward_market_research()
PY
```

Run the methodology audit:
```bash
python3 - <<'PY'
from src.train_baseline_models import run_modeling_audit
run_modeling_audit()
PY
```
