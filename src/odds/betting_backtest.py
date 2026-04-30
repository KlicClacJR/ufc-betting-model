"""Exploratory betting backtest helpers built on frozen model predictions."""

from src._pipeline_impl import (
    build_bet_selection_frame,
    build_betting_backtest_summary,
    compute_max_drawdown,
    filter_bets_for_threshold,
    prepare_model_backtest_dataframe,
    prepare_random_forest_backtest_dataframe,
    run_flat_betting_backtest,
    run_fractional_kelly_backtest,
    run_logistic_betting_backtest,
    run_betting_model_backtest_comparison,
    run_random_forest_betting_backtest,
    run_model_betting_backtest,
    select_best_flat_threshold,
    summarize_backtest_bets,
)

# Cleaner alias for the main post-cutoff betting workflow.
run_betting_backtest = run_logistic_betting_backtest
