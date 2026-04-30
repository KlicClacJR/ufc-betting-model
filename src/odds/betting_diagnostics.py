"""Post-hoc betting diagnostics and segment analysis."""

from src._pipeline_impl import (
    add_betting_diagnostic_buckets,
    build_betting_failure_diagnostics_summary,
    build_cumulative_profit_table,
    build_edge_correlation_table,
    build_edge_quality_table,
    build_favorite_longshot_analysis,
    build_probability_decile_calibration,
    build_time_segment_performance,
    run_logistic_betting_failure_diagnostics,
    run_betting_failure_diagnostics,
    run_model_betting_failure_diagnostics,
    run_random_forest_betting_failure_diagnostics,
    summarize_segment_groups,
)
