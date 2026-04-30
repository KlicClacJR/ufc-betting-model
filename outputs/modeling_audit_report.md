# UFC Betting Model Audit Report

- Audit date: 2026-04-28 21:06:29 EDT
- Overall status: PASS WITH MINOR ISSUES

## Overall Findings
- Final metrics reproducible from saved predictions: True
- Final model comparison valid: True
- Betting backtest calculations valid: True

## Critical Methodological Issues
- None found.

## Minor Issues
- Final logistic and Random Forest JSON artifacts do not explicitly store every code-injected default like random_state/n_jobs, even though the code path applies them consistently.
- The betting backtest reports the best threshold by test ROI; that ranking is descriptive only and should not be treated as a selected production rule.

## Split Verification
- Train/validation/test row overlap: {'train_validation_overlap_rows': 0, 'train_test_overlap_rows': 0, 'validation_test_overlap_rows': 0}
- Shared boundary dates: {'shared_dates_train_validation': 0, 'shared_dates_validation_test': 0, 'shared_dates_train_test': 0}
- Modeling-era cutoff enforced: fights on or after 2010-01-01.
- Interpretation: row leakage across splits was not detected, and split boundaries are grouped by event date.

## Feature Set Verification
- Frozen final feature set: ['diff_age', 'diff_total_wins', 'diff_total_losses', 'diff_win_streak', 'diff_td_landed_per_fight', 'diff_kd_absorbed_per_fight', 'diff_reach', 'diff_sig_strike_accuracy', 'diff_avg_fight_duration_seconds', 'diff_decision_rate', 'pre_fight_red_late_finish_rate', 'diff_five_round_experience']
- Modeling rows before 2010-01-01 are excluded because pre-2010 corner assignment appears inconsistent in the source data.
- All final features exist in `modeling_dataset_v2` and in the rebuilt modeling frame.
- Final logistic, Random Forest, and XGBoost evaluation artifacts all reference the same frozen feature set.
- No target, outcome, method, round, time, referee, details, or current-fight raw stat columns appear in the final frozen feature list.

## Preprocessing Verification
- Logistic regression uses a `Pipeline` with `SimpleImputer(strategy="median")` and `StandardScaler()`, and both are fit only on the training data used for that phase.
- Random Forest and XGBoost use training-only median imputation and do not scale features.
- Final hold-out evaluations fit preprocessing on train+validation only and transform test using those fitted objects.

## Model Configuration Verification
- Logistic final config resolves to elastic net with `solver='saga'`, `C=10.0`, `l1_ratio=0.8`, and `random_state=42` injected in code.
- Random Forest final config uses the validation-selected params plus `random_state=42` and `n_jobs=-1` injected in code.
- XGBoost final config matches the validation-selected best parameter payload directly.

## Hyperparameter Selection Verification
- Random Forest validation-best params match frozen saved params: True
- Random Forest validation results contain test columns: False
- XGBoost validation-best params match frozen saved params: True
- XGBoost validation results contain test columns: False
- No test metric columns were found in the RF or XGBoost validation-search result tables.

## Metric Definition Verification
- ROC AUC is computed from true labels and predicted probabilities.
- Log loss is computed from predicted probabilities via `sklearn.metrics.log_loss`; scikit-learn applies internal clipping for numerical stability.
- Brier score is computed as mean squared error between predicted probability and binary outcome.
- Accuracy, precision, recall, and F1 are computed using a 0.5 probability threshold.

## Prediction File Checks
- All three final prediction files have the expected test-set row count and probabilities within [0, 1].
- Saved hard labels match thresholding the saved probabilities at 0.5.

## Why Results Changed During the Project
- The modeling universe now excludes fights before 2010-01-01 because early-year red-corner win rates indicate a corner-assignment regime break.
- Validation metrics and final hold-out test metrics come from different datasets, so ranking can change even when code is correct.
- Early stages compared baseline models, later stages compared tuned or frozen-final versions.
- Most experimentation used train → validation, while final evaluation used train+validation → test.
- Different metrics reward different behavior: Random Forest won test log loss/Brier, while logistic won test ROC AUC.
- The betting backtest uses only the odds-matched subset of the hold-out test fights, not the full test set, so its behavior can diverge from overall test metrics.

## Betting Backtest Audit
- Betting backtest used the frozen Random Forest test predictions from `outputs/final_random_forest_test_predictions.csv`.
- Odds merge did not retrain any model; it only aligned existing predictions to the matched odds subset.
- Edge was computed against no-vig implied probabilities.
- Flat-bet profit formula is correct for decimal odds: profit = odds - 1 on wins, -1 on losses.
- Kelly negative bets found: 0
- Kelly cap violations found: 0
- Best flat threshold audited for sample checks: 0.08

### Manual Sample Check (5 bets)
| event_date | red_fighter_name | blue_fighter_name | selected_side | selected_decimal_odds | selected_model_prob | selected_market_prob_novig | selected_edge | recalc_edge | edge_difference | bet_won | profit | recalc_profit | profit_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01-13 00:00:00 | ANDREI ARLOVSKI | WALDO CORTES ACOSTA | red | 5.0 | 0.4376524014338128 | 0.1884816753926701 | 0.2491707260411426 | 0.24917072604114268 | -8.326672684688674e-17 | False | -1.0 | -1.0 | 0.0 |
| 2024-01-13 00:00:00 | FARID BASHARAT | TAYLOR LAPILUS | blue | 3.7 | 0.4527923626038788 | 0.2550335570469798 | 0.197758805556899 | 0.197758805556899 | 0.0 | False | -1.0 | -1.0 | 0.0 |
| 2024-01-13 00:00:00 | MAGOMED ANKALAEV | JOHNNY WALKER | blue | 4.9 | 0.5180318307635667 | 0.1957070707070707 | 0.322324760056496 | 0.322324760056496 | 0.0 | False | -1.0 | -1.0 | 0.0 |
| 2024-01-13 00:00:00 | NIKOLAS MOTTA | TOM NOLAN | red | 3.85 | 0.4616031723040808 | 0.2491874322860238 | 0.212415740018057 | 0.21241574001805702 | -2.7755575615628914e-17 | True | 2.85 | 2.85 | 0.0 |
| 2024-01-13 00:00:00 | WESTIN WILSON | JEAN SILVA | red | 7.25 | 0.5087104196187533 | 0.1322834645669291 | 0.3764269550518242 | 0.3764269550518242 | 0.0 | False | -1.0 | -1.0 | 0.0 |

## Notebook Consistency Check
- Markdown-only wording fixes were applied to notebooks 03, 04, and 05 so they no longer imply that the hold-out test is untouched after the later final benchmark sections.
- No code cells or metric outputs were changed by the notebook cleanup.

## Recommended Cleanup Before GitHub
- Freeze the final benchmark path around `modeling_dataset_v2` explicitly, even though the current rebuilt frame matches it exactly.
- Keep validation-selected strategy rules separate from descriptive test-set backtest summaries.

## Trust Verdict
- Final saved model metrics should be trusted as frozen benchmark outputs.
- The final betting backtest should be treated as exploratory diagnostics rather than a production betting strategy.