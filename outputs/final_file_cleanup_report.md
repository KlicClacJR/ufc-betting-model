# Final File Cleanup Report

## KEEP_IN_REPO
- outputs/final_model_test_comparison.csv
- outputs/final_model_comparison_summary.txt
- outputs/final_logistic_summary.txt
- outputs/final_random_forest_summary.txt
- outputs/final_xgboost_summary.txt
- outputs/final_logistic_test_metrics.json
- outputs/final_random_forest_test_metrics.json
- outputs/final_xgboost_test_metrics.json
- outputs/final_random_forest_feature_importance.csv
- outputs/final_xgboost_feature_importance.csv
- outputs/final_logistic_coefficients.csv
- outputs/modeling_audit_report.md
- outputs/final_github_readiness_check.md
- outputs/final_file_cleanup_report.md
- outputs/betting_backtest_summary.txt
- outputs/betting_failure_diagnostics_summary.txt
- outputs/odds_merge_diagnostics.txt
- outputs/selected_odds_source.txt
- outputs/odds_source_coverage.csv
- outputs/figures/

## ARCHIVE
- outputs/archive/baseline_metrics.json
- outputs/archive/best_elo_logistic_coefficients.csv
- outputs/archive/best_hypothesis_logistic_coefficients.csv
- outputs/archive/best_logistic_coefficients.csv
- outputs/archive/best_logistic_coefficients_v2.csv
- outputs/archive/best_model_formula.txt
- outputs/archive/best_model_formula_v2.txt
- outputs/archive/best_recency_logistic_coefficients.csv
- outputs/archive/cardio_feature_leave_one_out.csv
- outputs/archive/cardio_feature_subset_comparison.csv
- outputs/archive/cardio_feature_summary.txt
- outputs/archive/elo_k_validation_results.csv
- outputs/archive/elo_model_comparison.csv
- outputs/archive/elo_model_summary.txt
- outputs/archive/exp_decay_recency_validation_results.csv
- outputs/archive/hypothesis_feature_group_ablation.csv
- outputs/archive/hypothesis_feature_group_comparison.csv
- outputs/archive/hypothesis_feature_summary.txt
- outputs/archive/logistic_coefficients.csv
- outputs/archive/recency_model_comparison.csv
- outputs/archive/recency_model_summary.txt
- outputs/archive/validation_model_summary_v2.txt

## GITIGNORE
- data/UFC_betting_odds.csv
- data/modeling_dataset_with_odds.csv
- data/combined_statistics.csv
- data/modeling_dataset_v1.csv
- data/modeling_dataset_v2.csv
- outputs/*_predictions.csv
- outputs/*_bets.csv
- outputs/archive/
- __pycache__/
- .ipynb_checkpoints/
- .DS_Store

Notes:
- Archived files were moved rather than deleted.
- Prediction CSVs and bet-level logs remain on disk for reproducibility but are ignored for GitHub publication.