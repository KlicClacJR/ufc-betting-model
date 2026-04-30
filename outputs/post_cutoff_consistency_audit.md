# Post-Cutoff Consistency Audit

- Audit date: 2026-04-29
- Overall verdict: PASS WITH MINOR ISSUES

## Dataset Scope
- Raw combined dataset rows: 8591
- Raw combined date range: 1994-03-11 to 2026-03-28
- Modeling dataset rows after cutoff: 7347
- Modeling dataset date range: 2010-01-02 to 2026-03-28
- Modeling dataset with odds rows: 7347
- Modeling dataset with odds date range: 2010-01-02 to 2026-03-28
- Pre-2010 rows in modeling_dataset_v2: 0
- Pre-2010 rows in modeling_dataset_with_odds: 0

## Split Summary
- Train: 5057 rows, 2010-01-02 to 2021-10-16, red_win mean=0.575835
- Validation: 1135 rows, 2021-10-23 to 2023-12-16, red_win mean=0.570925
- Test: 1155 rows, 2024-01-13 to 2026-03-28, red_win mean=0.543723
- Split boundaries use unique event dates: True

## Final Model Winners After Cutoff
- Best log loss: logistic_regression
- Best ROC AUC: logistic_regression
- Best Brier score: logistic_regression
- Best accuracy: random_forest

## Betting / Walk-Forward Scope
- Betting matched test fight count: 367
- Betting min/max date: 2024-01-13 to 2024-12-07
- Walk-forward fold count: 11
- Walk-forward fold min/max test years: 2013 to 2023

## Stale Claims Found And Fixed
- README final comparison bullets were updated to reflect logistic regression as the post-cutoff probability winner and Random Forest as the post-cutoff accuracy winner.
- Saved rendered outputs in `04_random_forest.ipynb` and `05_xgboost.ipynb` were refreshed so they no longer display old pre-cutoff final comparison metrics.
- The final comparison text artifacts were checked to ensure no stale line still says Random Forest is the best probability model after the cutoff.

## Remaining Concerns
- The betting backtest still uses frozen Random Forest probabilities for continuity, even though logistic regression is now the strongest final post-cutoff probability model on the consumed hold-out benchmark.
- The betting threshold ranked by hold-out ROI remains descriptive only and should not be treated as a selected production rule.
- The journal notebook was intentionally left untouched.