"""Exploratory betting backtest helpers built on frozen model predictions."""

from src._pipeline_impl import (
    build_bet_selection_frame,
    build_betting_backtest_summary,
    compute_max_drawdown,
    filter_bets_for_threshold,
    prepare_random_forest_backtest_dataframe,
    run_flat_betting_backtest,
    run_fractional_kelly_backtest,
    run_random_forest_betting_backtest,
    select_best_flat_threshold,
    summarize_backtest_bets,
)

# Cleaner alias for publication-facing imports.
run_betting_backtest = run_random_forest_betting_backtest
