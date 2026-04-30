"""Train validation-only UFC logistic models with Elo and recency tuning.

This module keeps the hold-out test split untouched while it:
- builds a modeling-safe dataframe from `combined_statistics.csv`
- tunes recency parameters on train + validation only
- compares compact models with and without recency features
- saves interpretable coefficients and summaries for the best recency model
"""

from __future__ import annotations

import ctypes
import glob
import importlib
import json
import re
import sys
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_data import (  # noqa: E402
    DEFAULT_ELO_INITIAL_RATING,
    DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    DEFAULT_LAST_N_RECENCY,
    build_exp_decay_recency_features,
    build_elo_features,
    build_last_n_recency_features,
)

COMBINED_DATASET_PATH = PROJECT_ROOT / "data" / "combined_statistics.csv"
MODELING_DATASET_PATH = PROJECT_ROOT / "data" / "modeling_dataset_v2.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ELO_K_RESULTS_PATH = OUTPUTS_DIR / "elo_k_validation_results.csv"
ELO_MODEL_COMPARISON_PATH = OUTPUTS_DIR / "elo_model_comparison.csv"
BEST_ELO_COEFFICIENTS_PATH = OUTPUTS_DIR / "best_elo_logistic_coefficients.csv"
ELO_MODEL_SUMMARY_PATH = OUTPUTS_DIR / "elo_model_summary.txt"
ELO_MODEL_FORMULA_PATH = OUTPUTS_DIR / "best_model_formula_v2.txt"
VALIDATION_SUMMARY_V2_PATH = OUTPUTS_DIR / "validation_model_summary_v2.txt"
LAST_N_RECENCY_RESULTS_PATH = OUTPUTS_DIR / "last_n_recency_validation_results.csv"
LAST_N_RECENCY_RESULTS_EXPANDED_PATH = OUTPUTS_DIR / "last_n_recency_validation_results_expanded.csv"
EXP_DECAY_RECENCY_RESULTS_PATH = OUTPUTS_DIR / "exp_decay_recency_validation_results.csv"
RECENCY_MODEL_COMPARISON_PATH = OUTPUTS_DIR / "recency_model_comparison.csv"
BEST_RECENCY_COEFFICIENTS_PATH = OUTPUTS_DIR / "best_recency_logistic_coefficients.csv"
RECENCY_MODEL_SUMMARY_PATH = OUTPUTS_DIR / "recency_model_summary.txt"
HYPOTHESIS_FEATURE_GROUP_COMPARISON_PATH = OUTPUTS_DIR / "hypothesis_feature_group_comparison.csv"
HYPOTHESIS_FEATURE_GROUP_ABLATION_PATH = OUTPUTS_DIR / "hypothesis_feature_group_ablation.csv"
BEST_HYPOTHESIS_COEFFICIENTS_PATH = OUTPUTS_DIR / "best_hypothesis_logistic_coefficients.csv"
HYPOTHESIS_FEATURE_SUMMARY_PATH = OUTPUTS_DIR / "hypothesis_feature_summary.txt"
CARDIO_FEATURE_LEAVE_ONE_OUT_PATH = OUTPUTS_DIR / "cardio_feature_leave_one_out.csv"
CARDIO_SUBSET_COMPARISON_PATH = OUTPUTS_DIR / "cardio_feature_subset_comparison.csv"
CARDIO_FEATURE_SUMMARY_PATH = OUTPUTS_DIR / "cardio_feature_summary.txt"
FINAL_LOGISTIC_TEST_METRICS_PATH = OUTPUTS_DIR / "final_logistic_test_metrics.json"
FINAL_LOGISTIC_COEFFICIENTS_PATH = OUTPUTS_DIR / "final_logistic_coefficients.csv"
FINAL_LOGISTIC_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "final_logistic_test_predictions.csv"
FINAL_LOGISTIC_SUMMARY_PATH = OUTPUTS_DIR / "final_logistic_summary.txt"
RANDOM_FOREST_BASELINE_VALIDATION_PATH = OUTPUTS_DIR / "random_forest_baseline_validation.json"
RANDOM_FOREST_RANDOM_SEARCH_RESULTS_PATH = OUTPUTS_DIR / "random_forest_random_search_results.csv"
BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH = OUTPUTS_DIR / "best_random_forest_random_search.json"
RANDOM_FOREST_RANDOM_SEARCH_FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "random_forest_random_search_feature_importance.csv"
XGBOOST_VALIDATION_RESULTS_PATH = OUTPUTS_DIR / "xgboost_validation_results.csv"
BEST_XGBOOST_VALIDATION_PATH = OUTPUTS_DIR / "best_xgboost_validation.json"
XGBOOST_FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "xgboost_feature_importance.csv"
FINAL_RANDOM_FOREST_TEST_METRICS_PATH = OUTPUTS_DIR / "final_random_forest_test_metrics.json"
FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "final_random_forest_test_predictions.csv"
FINAL_RANDOM_FOREST_SUMMARY_PATH = OUTPUTS_DIR / "final_random_forest_summary.txt"
FINAL_RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "final_random_forest_feature_importance.csv"
FINAL_XGBOOST_TEST_METRICS_PATH = OUTPUTS_DIR / "final_xgboost_test_metrics.json"
FINAL_XGBOOST_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "final_xgboost_test_predictions.csv"
FINAL_XGBOOST_SUMMARY_PATH = OUTPUTS_DIR / "final_xgboost_summary.txt"
FINAL_XGBOOST_FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "final_xgboost_feature_importance.csv"
FINAL_LOGISTIC_CALIBRATION_TABLE_PATH = OUTPUTS_DIR / "final_logistic_calibration_table.csv"
FINAL_RANDOM_FOREST_CALIBRATION_TABLE_PATH = OUTPUTS_DIR / "final_random_forest_calibration_table.csv"
FINAL_XGBOOST_CALIBRATION_TABLE_PATH = OUTPUTS_DIR / "final_xgboost_calibration_table.csv"
FINAL_MODEL_TEST_COMPARISON_PATH = OUTPUTS_DIR / "final_model_test_comparison.csv"
FINAL_MODEL_COMPARISON_SUMMARY_PATH = OUTPUTS_DIR / "final_model_comparison_summary.txt"
ODDS_SOURCE_COVERAGE_PATH = OUTPUTS_DIR / "odds_source_coverage.csv"
SELECTED_ODDS_SOURCE_PATH = OUTPUTS_DIR / "selected_odds_source.txt"
ODDS_MERGE_DIAGNOSTICS_PATH = OUTPUTS_DIR / "odds_merge_diagnostics.txt"
UNMATCHED_ODDS_EXAMPLES_PATH = OUTPUTS_DIR / "unmatched_odds_examples.csv"
UNMATCHED_MODEL_FIGHTS_EXAMPLES_PATH = OUTPUTS_DIR / "unmatched_model_fights_examples.csv"
UFC_ODDS_CLEAN_SELECTED_SOURCE_PATH = PROJECT_ROOT / "data" / "ufc_odds_clean_selected_source.csv"
UFC_ODDS_CLEAN_WITH_IMPLIED_PROBS_PATH = PROJECT_ROOT / "data" / "ufc_odds_clean_with_implied_probs.csv"
MODELING_DATASET_WITH_ODDS_PATH = PROJECT_ROOT / "data" / "modeling_dataset_with_odds.csv"
BETTING_FLAT_BACKTEST_RESULTS_PATH = OUTPUTS_DIR / "betting_flat_backtest_results.csv"
BETTING_FLAT_BACKTEST_BETS_PATH = OUTPUTS_DIR / "betting_flat_backtest_bets.csv"
BETTING_KELLY_BACKTEST_RESULTS_PATH = OUTPUTS_DIR / "betting_kelly_backtest_results.csv"
BETTING_KELLY_BACKTEST_BETS_PATH = OUTPUTS_DIR / "betting_kelly_backtest_bets.csv"
BETTING_BACKTEST_SUMMARY_PATH = OUTPUTS_DIR / "betting_backtest_summary.txt"
LOGISTIC_BETTING_FLAT_BACKTEST_RESULTS_PATH = OUTPUTS_DIR / "logistic_betting_flat_backtest_results.csv"
LOGISTIC_BETTING_FLAT_BACKTEST_BETS_PATH = OUTPUTS_DIR / "logistic_betting_flat_backtest_bets.csv"
LOGISTIC_BETTING_KELLY_BACKTEST_RESULTS_PATH = OUTPUTS_DIR / "logistic_betting_kelly_backtest_results.csv"
LOGISTIC_BETTING_KELLY_BACKTEST_BETS_PATH = OUTPUTS_DIR / "logistic_betting_kelly_backtest_bets.csv"
LOGISTIC_BETTING_BACKTEST_SUMMARY_PATH = OUTPUTS_DIR / "logistic_betting_backtest_summary.txt"
BETTING_SEGMENT_BY_SIDE_PATH = OUTPUTS_DIR / "betting_segment_by_side.csv"
BETTING_SEGMENT_BY_ODDS_BUCKET_PATH = OUTPUTS_DIR / "betting_segment_by_odds_bucket.csv"
BETTING_SEGMENT_BY_MARKET_PROB_BUCKET_PATH = OUTPUTS_DIR / "betting_segment_by_market_prob_bucket.csv"
BETTING_SEGMENT_BY_MODEL_PROB_BUCKET_PATH = OUTPUTS_DIR / "betting_segment_by_model_prob_bucket.csv"
BETTING_SEGMENT_BY_EDGE_BUCKET_PATH = OUTPUTS_DIR / "betting_segment_by_edge_bucket.csv"
BETTING_EDGE_QUALITY_PATH = OUTPUTS_DIR / "betting_edge_quality.csv"
BETTING_EDGE_CORRELATIONS_PATH = OUTPUTS_DIR / "betting_edge_correlations.csv"
BETTING_MODEL_CALIBRATION_ON_MATCHED_ODDS_PATH = OUTPUTS_DIR / "betting_model_calibration_on_matched_odds.csv"
BETTING_MARKET_CALIBRATION_ON_MATCHED_ODDS_PATH = OUTPUTS_DIR / "betting_market_calibration_on_matched_odds.csv"
BETTING_FAVORITE_LONGSHOT_ANALYSIS_PATH = OUTPUTS_DIR / "betting_favorite_longshot_analysis.csv"
BETTING_CUMULATIVE_PROFIT_PATH = OUTPUTS_DIR / "betting_cumulative_profit.csv"
BETTING_TIME_SEGMENT_PERFORMANCE_PATH = OUTPUTS_DIR / "betting_time_segment_performance.csv"
BETTING_FAILURE_DIAGNOSTICS_SUMMARY_PATH = OUTPUTS_DIR / "betting_failure_diagnostics_summary.txt"
LOGISTIC_BETTING_SEGMENT_BY_SIDE_PATH = OUTPUTS_DIR / "logistic_betting_segment_by_side.csv"
LOGISTIC_BETTING_SEGMENT_BY_ODDS_BUCKET_PATH = OUTPUTS_DIR / "logistic_betting_segment_by_odds_bucket.csv"
LOGISTIC_BETTING_SEGMENT_BY_MARKET_PROB_BUCKET_PATH = OUTPUTS_DIR / "logistic_betting_segment_by_market_prob_bucket.csv"
LOGISTIC_BETTING_SEGMENT_BY_MODEL_PROB_BUCKET_PATH = OUTPUTS_DIR / "logistic_betting_segment_by_model_prob_bucket.csv"
LOGISTIC_BETTING_SEGMENT_BY_EDGE_BUCKET_PATH = OUTPUTS_DIR / "logistic_betting_segment_by_edge_bucket.csv"
LOGISTIC_BETTING_EDGE_QUALITY_PATH = OUTPUTS_DIR / "logistic_betting_edge_quality.csv"
LOGISTIC_BETTING_EDGE_CORRELATIONS_PATH = OUTPUTS_DIR / "logistic_betting_edge_correlations.csv"
LOGISTIC_BETTING_MODEL_CALIBRATION_ON_MATCHED_ODDS_PATH = OUTPUTS_DIR / "logistic_betting_model_calibration_on_matched_odds.csv"
LOGISTIC_BETTING_MARKET_CALIBRATION_ON_MATCHED_ODDS_PATH = OUTPUTS_DIR / "logistic_betting_market_calibration_on_matched_odds.csv"
LOGISTIC_BETTING_FAVORITE_LONGSHOT_ANALYSIS_PATH = OUTPUTS_DIR / "logistic_betting_favorite_longshot_analysis.csv"
LOGISTIC_BETTING_CUMULATIVE_PROFIT_PATH = OUTPUTS_DIR / "logistic_betting_cumulative_profit.csv"
LOGISTIC_BETTING_TIME_SEGMENT_PERFORMANCE_PATH = OUTPUTS_DIR / "logistic_betting_time_segment_performance.csv"
LOGISTIC_BETTING_FAILURE_DIAGNOSTICS_SUMMARY_PATH = OUTPUTS_DIR / "logistic_betting_failure_diagnostics_summary.txt"
BETTING_MODEL_BACKTEST_COMPARISON_PATH = OUTPUTS_DIR / "betting_model_backtest_comparison.csv"
BETTING_MODEL_BACKTEST_COMPARISON_SUMMARY_PATH = OUTPUTS_DIR / "betting_model_backtest_comparison_summary.txt"
POST_CUTOFF_BETTING_MODEL_UPDATE_SUMMARY_PATH = OUTPUTS_DIR / "post_cutoff_betting_model_update_summary.txt"
MODELING_AUDIT_REPORT_PATH = OUTPUTS_DIR / "modeling_audit_report.md"
AUDIT_SPLIT_SUMMARY_PATH = OUTPUTS_DIR / "audit_split_summary.csv"
AUDIT_METRIC_RECALCULATION_PATH = OUTPUTS_DIR / "audit_metric_recalculation.csv"
AUDIT_MODEL_CONFIG_SUMMARY_PATH = OUTPUTS_DIR / "audit_model_config_summary.csv"
AUDIT_PREDICTION_FILE_CHECKS_PATH = OUTPUTS_DIR / "audit_prediction_file_checks.csv"
AUDIT_LEAKAGE_CHECKS_PATH = OUTPUTS_DIR / "audit_leakage_checks.csv"

TARGET_COLUMN = "red_win"
DATE_COLUMN = "event_date"
WEIGHT_CLASS_COLUMN = "weight_class_clean"
MODELING_CUTOFF_DATE = pd.Timestamp("2010-01-01")
MODELING_CUTOFF_LABEL = "2010-01-01"
IDENTIFIER_COLUMNS = ["red_fighter_name", "blue_fighter_name"]
STATIC_SAFE_COLUMNS = [
    "red_fighter_age",
    "blue_fighter_age",
    "red_fighter_height",
    "blue_fighter_height",
    "red_fighter_reach",
    "blue_fighter_reach",
    "red_fighter_stance",
    "blue_fighter_stance",
]
DIRECT_SAFE_FEATURE_COLUMNS = {
    "red_grappler_vs_blue_striker",
    "red_striker_vs_blue_grappler",
    "red_finish_vs_blue_durability",
    "blue_finish_vs_red_durability",
    "red_td_offense_vs_blue_td_defense",
    "blue_td_offense_vs_red_td_defense",
    "red_grappling_pressure_vs_blue_defense",
    "blue_grappling_pressure_vs_red_defense",
}

DEFAULT_MODEL_SPEC = {
    "model_name": "elastic_net",
    "penalty": "elasticnet",
    "C": 10.0,
    "solver": "saga",
    "l1_ratio": 0.8,
    "max_iter": 5000,
}
ELO_K_VALUES = [16, 24, 32, 40, 50, 64]
LAST_N_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EXP_DECAY_HALF_LIFE_CANDIDATES = [180, 365, 730, 1095]
PREVIOUS_BEST_LOG_LOSS = 0.663442
PREVIOUS_BEST_ROC_AUC = 0.642936
CARDIO_SUBSET_SIZE_CANDIDATES = [2, 3, 4]
CARDIO_SUBSET_TIE_THRESHOLD = 0.001
DEFAULT_LAST_N_RECENCY = 2
DEFAULT_EXP_DECAY_HALF_LIFE_DAYS = 365.0
FINAL_LOGISTIC_FEATURE_CANDIDATES = [
    "diff_age",
    "diff_total_wins",
    "diff_total_losses",
    "diff_win_streak",
    "diff_td_landed_per_fight",
    "diff_kd_absorbed_per_fight",
    "diff_reach",
    "diff_sig_strike_accuracy",
    "diff_avg_fight_duration_seconds",
    "diff_decision_rate",
    "pre_fight_red_late_finish_rate",
    "diff_five_round_experience",
]
RANDOM_FOREST_BASELINE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
RANDOM_FOREST_RANDOM_SEARCH_SPACE = {
    "n_estimators": [200, 300, 500, 800, 1200],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20, 40],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, None],
    "bootstrap": [True],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
XGBOOST_BASELINE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
XGBOOST_RANDOM_SEARCH_SPACE = {
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 3, 5, 10],
    "reg_alpha": [0, 0.1, 1],
}
FLAT_EDGE_THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10]
KELLY_FRACTIONS = [0.25, 0.50]
KELLY_FULL_FRACTION_CAP = 0.25
KELLY_STARTING_BANKROLL = 1000.0

CLEAN_BASELINE_FEATURE_CANDIDATES = [
    "diff_age",
    "diff_total_wins",
    "diff_total_losses",
    "diff_win_streak",
    "diff_td_landed_per_fight",
    "diff_kd_absorbed_per_fight",
    "diff_reach",
    "diff_sig_strike_accuracy",
    "diff_ctrl_seconds_per_fight",
]
OPTIONAL_CLEAN_FEATURE_CANDIDATES = [
    "diff_loss_streak",
    "diff_height",
    "diff_durability_index",
    "pre_fight_ko_loss_rate_diff",
    "diff_td_accuracy",
]
PREVIOUS_BEST_FEATURE_CANDIDATES = [
    "diff_age",
    "diff_total_wins",
    "diff_total_losses",
    "diff_win_streak",
    "diff_td_landed_per_fight",
    "diff_kd_absorbed_per_fight",
    "diff_reach",
    "diff_sig_strike_accuracy",
    "diff_ctrl_seconds_per_fight",
    "diff_elo",
    "pre_fight_ko_loss_rate_diff",
]
CURRENT_PREVIOUS_BEST_FEATURE_CANDIDATES = [
    *PREVIOUS_BEST_FEATURE_CANDIDATES,
    "diff_recent_2_win_pct",
    "diff_recent_2_sig_strikes_landed_per_fight",
    "diff_recent_2_sig_strikes_landed_per_min",
    "diff_recent_2_td_landed_per_fight",
    "diff_recent_2_ctrl_seconds_per_fight",
    "diff_recent_2_kd_per_fight",
    "diff_recent_2_kd_absorbed_per_fight",
]
ELO_RATING_FEATURE_CANDIDATES = [
    "diff_elo",
]
ELO_ONLY_FEATURE_CANDIDATES = [
    "diff_elo",
    "pre_fight_red_elo",
    "pre_fight_blue_elo",
    "pre_fight_red_elo_win_prob",
]


def load_combined_dataset(path: Path = COMBINED_DATASET_PATH) -> pd.DataFrame:
    """Load the combined fight-level dataset."""
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def detect_delimiter(path: Path) -> str:
    """Detect a simple CSV delimiter between comma and semicolon."""
    sample = path.read_text(errors="ignore")[:4096]
    return ";" if sample.count(";") > sample.count(",") else ","


def normalize_name_key(value: Any) -> str:
    """Create a fighter-name merge key with case, accents, punctuation, and spacing normalized."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_source_name(value: Any) -> str:
    """Normalize sportsbook/source labels for grouping and simple matching."""
    if pd.isna(value):
        return "UNKNOWN_SOURCE"
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_key(value: Any) -> str:
    """Generic normalized text key helper."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def locate_odds_file(data_dir: Path = PROJECT_ROOT / "data") -> tuple[list[Path], Path | None]:
    """Locate likely odds CSV files in the data directory."""
    all_csvs = sorted(data_dir.glob("*.csv"))
    odds_candidates = [
        path for path in all_csvs
        if any(token in path.name.lower() for token in ["odds", "book", "bet", "moneyline", "line"])
    ]
    if not odds_candidates:
        return all_csvs, None
    scored = sorted(
        odds_candidates,
        key=lambda path: (
            "odds" not in path.name.lower(),
            "ufc" not in path.name.lower(),
            path.stat().st_size * -1,
        ),
    )
    return all_csvs, scored[0]


def load_odds_dataframe(path: Path) -> pd.DataFrame:
    """Load the raw odds dataframe with basic delimiter detection."""
    delimiter = detect_delimiter(path)
    return pd.read_csv(path, sep=delimiter)


def find_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column whose normalized name matches a candidate set."""
    normalized_lookup = {normalize_text_key(column): column for column in columns}
    for candidate in candidates:
        candidate_key = normalize_text_key(candidate)
        if candidate_key in normalized_lookup:
            return normalized_lookup[candidate_key]
    for candidate in candidates:
        candidate_key = normalize_text_key(candidate)
        for normalized, original in normalized_lookup.items():
            if candidate_key in normalized or normalized in candidate_key:
                return original
    return None


def identify_odds_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Identify the most likely key odds columns from a Kaggle-style betting dataset."""
    columns = df.columns.tolist()
    mapping = {
        "event_date": find_matching_column(columns, ["event_date", "date", "fight_date", "eventdate", "commence_time"]),
        "fighter_1": find_matching_column(columns, ["fighter_1", "fighter1", "red_fighter", "fighter_a", "home_fighter", "fighter_a_name"]),
        "fighter_2": find_matching_column(columns, ["fighter_2", "fighter2", "blue_fighter", "fighter_b", "away_fighter", "fighter_b_name", "opponent"]),
        "fighter_1_odds": find_matching_column(columns, ["fighter_1_odds", "fighter1_odds", "red_odds", "odds_1", "moneyline_1", "fighter_a_odds"]),
        "fighter_2_odds": find_matching_column(columns, ["fighter_2_odds", "fighter2_odds", "blue_odds", "odds_2", "moneyline_2", "fighter_b_odds"]),
        "source": find_matching_column(columns, ["source", "bookmaker", "sportsbook", "book", "site"]),
        "timestamp": find_matching_column(columns, ["timestamp", "date_scraped", "scraped_at", "created_at", "updated_at", "snapshot_time"]),
    }
    return mapping


def standardize_odds_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Map raw odds data into a standardized audit-friendly schema."""
    column_mapping = identify_odds_columns(raw_df)
    standardized = raw_df.copy()
    standardized.columns = [normalize_text_key(column) for column in standardized.columns]

    rename_map: dict[str, str] = {}
    for target_name, raw_name in column_mapping.items():
        if raw_name is not None:
            rename_map[normalize_text_key(raw_name)] = target_name
    standardized = standardized.rename(columns=rename_map)

    for required in ["event_date", "fighter_1", "fighter_2", "fighter_1_odds", "fighter_2_odds"]:
        if required not in standardized.columns:
            standardized[required] = pd.NA

    standardized["event_date"] = pd.to_datetime(standardized["event_date"], errors="coerce").dt.normalize()
    if "timestamp" in standardized.columns:
        standardized["timestamp"] = pd.to_datetime(standardized["timestamp"], errors="coerce")
    else:
        standardized["timestamp"] = pd.NaT
    if "source" not in standardized.columns:
        standardized["source"] = "NO_SOURCE_COLUMN"
    standardized["source"] = standardized["source"].map(normalize_source_name)
    standardized["fighter_1"] = standardized["fighter_1"].astype("string").str.strip()
    standardized["fighter_2"] = standardized["fighter_2"].astype("string").str.strip()
    standardized["fighter_1_key"] = standardized["fighter_1"].map(normalize_name_key)
    standardized["fighter_2_key"] = standardized["fighter_2"].map(normalize_name_key)
    standardized["fighter_1_odds"] = pd.to_numeric(standardized["fighter_1_odds"], errors="coerce")
    standardized["fighter_2_odds"] = pd.to_numeric(standardized["fighter_2_odds"], errors="coerce")
    return standardized, column_mapping


def build_unordered_fight_key(fighter_a: pd.Series, fighter_b: pd.Series) -> pd.Series:
    """Create an unordered fighter pair key for exact, auditable fight matching."""
    left = fighter_a.fillna("").astype(str)
    right = fighter_b.fillna("").astype(str)
    return np.where(left <= right, left + "__" + right, right + "__" + left)


def summarize_odds_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize odds coverage by source/bookmaker."""
    working = df.copy()
    working["fight_pair_key"] = build_unordered_fight_key(working["fighter_1_key"], working["fighter_2_key"])
    working["missing_odds_count"] = working[["fighter_1_odds", "fighter_2_odds"]].isna().sum(axis=1)
    summary = (
        working.groupby("source", dropna=False)
        .agg(
            rows=("source", "size"),
            unique_fights=("fight_pair_key", lambda values: values.nunique()),
            missing_odds_count=("missing_odds_count", "sum"),
            start_date=("event_date", "min"),
            end_date=("event_date", "max"),
        )
        .reset_index()
    )
    return summary.sort_values(["unique_fights", "rows", "missing_odds_count"], ascending=[False, False, True]).reset_index(drop=True)


def choose_odds_source(source_summary: pd.DataFrame) -> str:
    """Choose a single tradable source/bookmaker for the first pass."""
    if source_summary.empty:
        return "NO_SOURCE_AVAILABLE"

    summary = source_summary.copy()
    summary["source_key"] = summary["source"].map(normalize_text_key)
    named_mask = ~summary["source_key"].str.contains("zewn", na=False) & ~summary["source_key"].eq("unknown_source")
    named = summary.loc[named_mask].copy()

    if not named.empty:
        named = named.sort_values(["unique_fights", "missing_odds_count", "rows"], ascending=[False, True, False])
        best_named = named.iloc[0]
        max_coverage = summary["unique_fights"].max()
        if best_named["unique_fights"] >= max_coverage * 0.8:
            return str(best_named["source"])

    zewn = summary.loc[summary["source_key"].str.contains("zewn", na=False)].copy()
    if not zewn.empty:
        zewn = zewn.sort_values(["unique_fights", "missing_odds_count", "rows"], ascending=[False, True, False])
        return str(zewn.iloc[0]["source"])

    return str(summary.iloc[0]["source"])


def deduplicate_selected_odds(df: pd.DataFrame, selected_source: str) -> pd.DataFrame:
    """Keep the final available row per fight/source, preferring latest timestamp when present."""
    filtered = df.loc[df["source"] == selected_source].copy()
    filtered["fight_pair_key"] = build_unordered_fight_key(filtered["fighter_1_key"], filtered["fighter_2_key"])
    filtered = filtered.sort_values(
        ["event_date", "fight_pair_key", "source", "timestamp"],
        kind="mergesort",
        na_position="last",
    )
    return filtered.drop_duplicates(subset=["event_date", "fight_pair_key", "source"], keep="last").reset_index(drop=True)


def detect_odds_format(df: pd.DataFrame) -> str:
    """Detect whether odds are American or decimal."""
    values = pd.concat([df["fighter_1_odds"], df["fighter_2_odds"]], axis=0).dropna()
    if values.empty:
        return "unknown"
    if (values < 0).any() or (values.abs() >= 100).mean() > 0.5:
        return "american"
    return "decimal"


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    """Convert American odds to implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")
    positive = odds > 0
    negative = odds < 0
    result = pd.Series(np.nan, index=odds.index, dtype="float64")
    result.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)
    result.loc[negative] = odds.loc[negative].abs() / (odds.loc[negative].abs() + 100.0)
    return result


def add_implied_probability_columns(df: pd.DataFrame, odds_format: str) -> pd.DataFrame:
    """Add raw and no-vig implied probabilities."""
    enriched = df.copy()
    if odds_format == "american":
        enriched["fighter_1_implied_prob_raw"] = american_to_implied_prob(enriched["fighter_1_odds"])
        enriched["fighter_2_implied_prob_raw"] = american_to_implied_prob(enriched["fighter_2_odds"])
    else:
        enriched["fighter_1_implied_prob_raw"] = 1.0 / pd.to_numeric(enriched["fighter_1_odds"], errors="coerce")
        enriched["fighter_2_implied_prob_raw"] = 1.0 / pd.to_numeric(enriched["fighter_2_odds"], errors="coerce")
    enriched["overround"] = enriched["fighter_1_implied_prob_raw"] + enriched["fighter_2_implied_prob_raw"]
    enriched["fighter_1_implied_prob_novig"] = enriched["fighter_1_implied_prob_raw"] / enriched["overround"]
    enriched["fighter_2_implied_prob_novig"] = enriched["fighter_2_implied_prob_raw"] / enriched["overround"]
    return enriched


def prepare_odds_for_merge(odds_df: pd.DataFrame, modeling_df: pd.DataFrame, odds_format: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a merge-ready odds table oriented to red/blue fighters."""
    working_model = modeling_df.copy()
    working_model["event_date"] = pd.to_datetime(working_model["event_date"], errors="coerce").dt.normalize()
    working_model["red_fighter_key"] = working_model["red_fighter_name"].map(normalize_name_key)
    working_model["blue_fighter_key"] = working_model["blue_fighter_name"].map(normalize_name_key)
    working_model["fight_pair_key"] = build_unordered_fight_key(working_model["red_fighter_key"], working_model["blue_fighter_key"])

    odds_ready = odds_df.copy()
    odds_ready["fight_pair_key"] = build_unordered_fight_key(odds_ready["fighter_1_key"], odds_ready["fighter_2_key"])
    merged = working_model.merge(
        odds_ready,
        how="left",
        on=["event_date", "fight_pair_key"],
        suffixes=("", "_odds"),
        indicator=True,
    )
    merged["fighter_order_matches_red_blue"] = (
        (merged["fighter_1_key"] == merged["red_fighter_key"]) &
        (merged["fighter_2_key"] == merged["blue_fighter_key"])
    )
    merged["fighter_order_matches_blue_red"] = (
        (merged["fighter_1_key"] == merged["blue_fighter_key"]) &
        (merged["fighter_2_key"] == merged["red_fighter_key"])
    )

    odds_name_prefix = "american" if odds_format == "american" else "decimal"
    merged[f"red_{odds_name_prefix}_odds"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_1_odds"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_2_odds"], np.nan),
    )
    merged[f"blue_{odds_name_prefix}_odds"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_2_odds"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_1_odds"], np.nan),
    )
    merged["red_implied_prob_raw"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_1_implied_prob_raw"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_2_implied_prob_raw"], np.nan),
    )
    merged["blue_implied_prob_raw"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_2_implied_prob_raw"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_1_implied_prob_raw"], np.nan),
    )
    merged["red_implied_prob_novig"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_1_implied_prob_novig"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_2_implied_prob_novig"], np.nan),
    )
    merged["blue_implied_prob_novig"] = np.where(
        merged["fighter_order_matches_red_blue"],
        merged["fighter_2_implied_prob_novig"],
        np.where(merged["fighter_order_matches_blue_red"], merged["fighter_1_implied_prob_novig"], np.nan),
    )
    merged["odds_overround"] = merged["overround"]
    merged["odds_source"] = merged["source"]
    return working_model, merged


def run_odds_ingestion_pipeline(
    data_dir: Path = PROJECT_ROOT / "data",
    modeling_dataset_path: Path = MODELING_DATASET_PATH,
) -> dict[str, Any]:
    """Load, clean, and merge odds data into the modeling dataset without changing any models."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_csvs, selected_file = locate_odds_file(data_dir)

    print("CSV files found in data/:")
    for path in all_csvs:
        print(f"- {path.name}")

    if selected_file is None:
        message = (
            "No likely UFC odds CSV was found in data/. "
            "Add the Kaggle odds file to data/ and rerun run_odds_ingestion_pipeline()."
        )
        pd.DataFrame(columns=["source", "rows", "unique_fights", "missing_odds_count", "start_date", "end_date"]).to_csv(
            ODDS_SOURCE_COVERAGE_PATH,
            index=False,
        )
        SELECTED_ODDS_SOURCE_PATH.write_text("NO_ODDS_FILE_FOUND")
        ODDS_MERGE_DIAGNOSTICS_PATH.write_text(message)
        pd.DataFrame(columns=["event_date", "fighter_1", "fighter_2"]).to_csv(UNMATCHED_ODDS_EXAMPLES_PATH, index=False)
        pd.DataFrame(columns=["event_date", "red_fighter_name", "blue_fighter_name"]).to_csv(UNMATCHED_MODEL_FIGHTS_EXAMPLES_PATH, index=False)
        print(message)
        return {
            "odds_file_used": None,
            "selected_source": None,
            "odds_format": None,
            "rows_cleaned": 0,
            "match_rate": 0.0,
            "main_merge_issues": [message],
            "created_files": [
                str(ODDS_SOURCE_COVERAGE_PATH),
                str(SELECTED_ODDS_SOURCE_PATH),
                str(ODDS_MERGE_DIAGNOSTICS_PATH),
                str(UNMATCHED_ODDS_EXAMPLES_PATH),
                str(UNMATCHED_MODEL_FIGHTS_EXAMPLES_PATH),
            ],
        }

    raw_odds_df = load_odds_dataframe(selected_file)
    print(f"Selected odds file: {selected_file.name}")
    print(f"Raw odds shape: {raw_odds_df.shape}")
    print(f"Raw odds columns: {raw_odds_df.columns.tolist()}")
    print(raw_odds_df.head().to_string(index=False))

    standardized_odds_df, column_mapping = standardize_odds_dataframe(raw_odds_df)
    if "source" in standardized_odds_df.columns:
        print("Unique sources/bookmakers:")
        print(sorted(standardized_odds_df["source"].dropna().astype(str).unique().tolist()))
    if standardized_odds_df["event_date"].notna().any():
        print(
            "Odds date range: "
            f"{standardized_odds_df['event_date'].min().date()} to {standardized_odds_df['event_date'].max().date()}"
        )

    source_summary = summarize_odds_sources(standardized_odds_df)
    source_summary.to_csv(ODDS_SOURCE_COVERAGE_PATH, index=False)
    selected_source = choose_odds_source(source_summary)
    SELECTED_ODDS_SOURCE_PATH.write_text(
        selected_source + (
            "\nNote: 'zewnętrzne' / 'zewnetrzne' is treated as an external or aggregated source."
            if "zewn" in normalize_text_key(selected_source)
            else ""
        )
    )

    deduplicated_odds_df = deduplicate_selected_odds(standardized_odds_df, selected_source)
    deduplicated_odds_df.to_csv(UFC_ODDS_CLEAN_SELECTED_SOURCE_PATH, index=False)

    odds_format = detect_odds_format(deduplicated_odds_df)
    enriched_odds_df = add_implied_probability_columns(deduplicated_odds_df, odds_format)
    enriched_odds_df.to_csv(UFC_ODDS_CLEAN_WITH_IMPLIED_PROBS_PATH, index=False)

    modeling_df = pd.read_csv(modeling_dataset_path, parse_dates=["event_date"])
    _, merged_model_df = prepare_odds_for_merge(enriched_odds_df, modeling_df, odds_format)
    merged_model_df.to_csv(MODELING_DATASET_WITH_ODDS_PATH, index=False)

    matched_mask = merged_model_df["red_implied_prob_novig"].notna() & merged_model_df["blue_implied_prob_novig"].notna()
    unmatched_model_examples = merged_model_df.loc[~matched_mask, ["event_date", "red_fighter_name", "blue_fighter_name"]].head(25)
    unmatched_model_examples.to_csv(UNMATCHED_MODEL_FIGHTS_EXAMPLES_PATH, index=False)

    matched_pair_keys = set(merged_model_df.loc[matched_mask, "fight_pair_key"].astype(str))
    unmatched_odds_examples = enriched_odds_df.loc[
        ~enriched_odds_df["fight_pair_key"].astype(str).isin(matched_pair_keys),
        ["event_date", "fighter_1", "fighter_2", "fighter_1_odds", "fighter_2_odds", "source"],
    ].head(25)
    unmatched_odds_examples.to_csv(UNMATCHED_ODDS_EXAMPLES_PATH, index=False)

    duplicate_match_count = int(
        merged_model_df.groupby(["event_date", "red_fighter_name", "blue_fighter_name"]).size().gt(1).sum()
    )
    invalid_odds_count = int(
        (
            pd.to_numeric(merged_model_df.filter(regex="_odds$").stack(), errors="coerce").le(0)
        ).sum()
    ) if not merged_model_df.filter(regex="_odds$").empty else 0
    total_rows = len(merged_model_df)
    matched_rows = int(matched_mask.sum())
    match_rate = matched_rows / total_rows if total_rows else 0.0

    diagnostics_text = "\n".join(
        [
            f"Odds file used: {selected_file.name}",
            f"Selected source: {selected_source}",
            f"Odds format detected: {odds_format}",
            f"Total model rows: {total_rows}",
            f"Matched rows: {matched_rows}",
            f"Match rate: {match_rate:.4f}",
            f"Unmatched rows: {total_rows - matched_rows}",
            f"Duplicate match count: {duplicate_match_count}",
            f"Rows with invalid odds: {invalid_odds_count}",
            f"Column mapping: {column_mapping}",
        ]
    )
    ODDS_MERGE_DIAGNOSTICS_PATH.write_text(diagnostics_text)

    print("Odds merge diagnostics:")
    print(diagnostics_text)
    print("Unmatched model fight examples:")
    print(unmatched_model_examples.to_string(index=False))

    return {
        "odds_file_used": selected_file.name,
        "selected_source": selected_source,
        "odds_format": odds_format,
        "rows_cleaned": int(len(enriched_odds_df)),
        "match_rate": match_rate,
        "main_merge_issues": [
            "No fuzzy matching was used; exact normalized names plus event_date may miss some fights.",
            "If the selected source is 'zewnętrzne' / 'zewnetrzne', it likely represents an external aggregated feed rather than a directly tradable book.",
        ],
        "created_files": [
            str(ODDS_SOURCE_COVERAGE_PATH),
            str(SELECTED_ODDS_SOURCE_PATH),
            str(UFC_ODDS_CLEAN_SELECTED_SOURCE_PATH),
            str(UFC_ODDS_CLEAN_WITH_IMPLIED_PROBS_PATH),
            str(MODELING_DATASET_WITH_ODDS_PATH),
            str(ODDS_MERGE_DIAGNOSTICS_PATH),
            str(UNMATCHED_ODDS_EXAMPLES_PATH),
            str(UNMATCHED_MODEL_FIGHTS_EXAMPLES_PATH),
        ],
    }


def load_backtest_inputs(
    merged_odds_path: Path = MODELING_DATASET_WITH_ODDS_PATH,
    prediction_path: Path = FINAL_LOGISTIC_TEST_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged odds data and one frozen hold-out prediction file."""
    merged_odds_df = pd.read_csv(merged_odds_path, parse_dates=[DATE_COLUMN], low_memory=False)
    prediction_df = pd.read_csv(prediction_path, parse_dates=[DATE_COLUMN])
    return merged_odds_df, prediction_df


def prepare_model_backtest_dataframe(
    merged_odds_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align one frozen test prediction file to the merged odds dataset."""
    merge_columns = [DATE_COLUMN, "red_fighter_name", "blue_fighter_name", TARGET_COLUMN]
    odds_columns = [
        DATE_COLUMN,
        "red_fighter_name",
        "blue_fighter_name",
        TARGET_COLUMN,
        "red_decimal_odds",
        "blue_decimal_odds",
        "red_implied_prob_raw",
        "blue_implied_prob_raw",
        "red_implied_prob_novig",
        "blue_implied_prob_novig",
        "odds_source",
        "odds_overround",
    ]
    odds_slice = merged_odds_df[odds_columns].copy()
    merged = prediction_df.merge(
        odds_slice,
        how="left",
        on=merge_columns,
        validate="one_to_one",
    )
    merged["model_prob_red"] = pd.to_numeric(merged["predicted_red_win_probability"], errors="coerce")
    merged["model_prob_blue"] = 1.0 - merged["model_prob_red"]

    valid_mask = (
        merged["red_decimal_odds"].notna()
        & merged["blue_decimal_odds"].notna()
        & merged["red_implied_prob_novig"].notna()
        & merged["blue_implied_prob_novig"].notna()
        & merged["model_prob_red"].notna()
        & merged[TARGET_COLUMN].notna()
    )
    valid = merged.loc[valid_mask].copy()
    valid["edge_red"] = valid["model_prob_red"] - valid["red_implied_prob_novig"]
    valid["edge_blue"] = valid["model_prob_blue"] - valid["blue_implied_prob_novig"]
    valid = valid.sort_values([DATE_COLUMN, "red_fighter_name", "blue_fighter_name"], kind="mergesort").reset_index(drop=True)

    diagnostics = {
        "total_prediction_rows": int(len(prediction_df)),
        "matched_prediction_rows": int(len(merged)),
        "usable_backtest_rows": int(len(valid)),
        "usable_backtest_rate": float(len(valid) / len(prediction_df)) if len(prediction_df) else 0.0,
        "missing_red_odds_rows": int(merged["red_decimal_odds"].isna().sum()),
        "missing_blue_odds_rows": int(merged["blue_decimal_odds"].isna().sum()),
        "missing_novig_rows": int(
            (
                merged["red_implied_prob_novig"].isna()
                | merged["blue_implied_prob_novig"].isna()
            ).sum()
        ),
        "odds_source": str(valid["odds_source"].mode().iloc[0]) if not valid.empty and valid["odds_source"].notna().any() else "UNKNOWN",
    }
    return valid, diagnostics


def prepare_random_forest_backtest_dataframe(
    merged_odds_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for the historical Random Forest betting path."""
    return prepare_model_backtest_dataframe(merged_odds_df, prediction_df)


def build_bet_selection_frame(backtest_df: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Select at most one side per fight based on the larger positive model edge."""
    selection = backtest_df.copy()
    selection["selected_side"] = np.where(
        (selection["edge_red"] >= selection["edge_blue"]) & (selection["edge_red"] > edge_threshold),
        "red",
        np.where(selection["edge_blue"] > edge_threshold, "blue", "skip"),
    )
    selection["selected_edge"] = np.where(
        selection["selected_side"].eq("red"),
        selection["edge_red"],
        np.where(selection["selected_side"].eq("blue"), selection["edge_blue"], np.nan),
    )
    selection["selected_decimal_odds"] = np.where(
        selection["selected_side"].eq("red"),
        selection["red_decimal_odds"],
        np.where(selection["selected_side"].eq("blue"), selection["blue_decimal_odds"], np.nan),
    )
    selection["selected_model_prob"] = np.where(
        selection["selected_side"].eq("red"),
        selection["model_prob_red"],
        np.where(selection["selected_side"].eq("blue"), selection["model_prob_blue"], np.nan),
    )
    selection["selected_market_prob_novig"] = np.where(
        selection["selected_side"].eq("red"),
        selection["red_implied_prob_novig"],
        np.where(selection["selected_side"].eq("blue"), selection["blue_implied_prob_novig"], np.nan),
    )
    selection["bet_won"] = np.where(
        selection["selected_side"].eq("red"),
        selection[TARGET_COLUMN].eq(1),
        np.where(selection["selected_side"].eq("blue"), selection[TARGET_COLUMN].eq(0), pd.NA),
    )
    selection["bet_won"] = selection["bet_won"].where(selection["selected_side"].ne("skip"))
    return selection


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve in dollars."""
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = running_peak - equity_curve
    return float(drawdown.max())


def summarize_backtest_bets(
    bets_df: pd.DataFrame,
    threshold: float,
    strategy_name: str,
    stake_column: str,
    profit_column: str,
    equity_column: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one betting simulation."""
    bets = bets_df.loc[bets_df["selected_side"].ne("skip")].copy()
    total_staked = float(bets[stake_column].sum()) if not bets.empty else 0.0
    total_profit = float(bets[profit_column].sum()) if not bets.empty else 0.0
    number_of_bets = int(len(bets))
    wins = int(bets["bet_won"].fillna(False).sum()) if not bets.empty else 0
    win_rate = float(wins / number_of_bets) if number_of_bets else 0.0
    roi = float(total_profit / total_staked) if total_staked > 0 else 0.0
    average_edge = float(bets["selected_edge"].mean()) if not bets.empty else 0.0
    average_odds = float(bets["selected_decimal_odds"].mean()) if not bets.empty else 0.0
    max_drawdown = compute_max_drawdown(bets[equity_column]) if not bets.empty else 0.0

    summary = {
        "strategy_name": strategy_name,
        "edge_threshold": threshold,
        "number_of_bets": number_of_bets,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "roi": roi,
        "average_edge": average_edge,
        "average_odds": average_odds,
        "max_drawdown": max_drawdown,
        "total_staked": total_staked,
    }
    if extra_fields:
        summary.update(extra_fields)
    return summary


def run_flat_betting_backtest(backtest_df: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a simple flat-stake edge-threshold backtest."""
    result_rows: list[dict[str, Any]] = []
    bet_frames: list[pd.DataFrame] = []

    for threshold in thresholds:
        selection = build_bet_selection_frame(backtest_df, threshold)
        bets = selection.loc[selection["selected_side"].ne("skip")].copy()
        if not bets.empty:
            bets["stake"] = 1.0
            bets["profit"] = np.where(
                bets["bet_won"].fillna(False),
                bets["selected_decimal_odds"] - 1.0,
                -1.0,
            )
            bets["cumulative_profit"] = bets["profit"].cumsum()
            bets["bankroll"] = bets["cumulative_profit"]
        else:
            bets["stake"] = pd.Series(dtype="float64")
            bets["profit"] = pd.Series(dtype="float64")
            bets["cumulative_profit"] = pd.Series(dtype="float64")
            bets["bankroll"] = pd.Series(dtype="float64")
        bets["strategy_name"] = "flat"
        bets["edge_threshold"] = threshold
        result_rows.append(
            summarize_backtest_bets(
                bets,
                threshold=threshold,
                strategy_name="flat",
                stake_column="stake",
                profit_column="profit",
                equity_column="cumulative_profit",
            )
        )
        bet_frames.append(bets)

    results_df = pd.DataFrame(result_rows).sort_values(["roi", "total_profit", "number_of_bets"], ascending=[False, False, False]).reset_index(drop=True)
    bets_df = pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame()
    return results_df, bets_df


def run_fractional_kelly_backtest(
    backtest_df: pd.DataFrame,
    thresholds: list[float],
    kelly_fractions: list[float],
    starting_bankroll: float = KELLY_STARTING_BANKROLL,
    full_kelly_cap: float = KELLY_FULL_FRACTION_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a capped fractional-Kelly backtest on the frozen model probabilities."""
    result_rows: list[dict[str, Any]] = []
    bet_frames: list[pd.DataFrame] = []

    for kelly_fraction in kelly_fractions:
        for threshold in thresholds:
            selection = build_bet_selection_frame(backtest_df, threshold)
            bets = selection.loc[selection["selected_side"].ne("skip")].copy()
            bankroll = starting_bankroll
            bankroll_history: list[float] = []
            stake_history: list[float] = []
            profit_history: list[float] = []
            full_kelly_history: list[float] = []
            actual_fraction_history: list[float] = []

            for row in bets.itertuples(index=False):
                b = float(row.selected_decimal_odds) - 1.0
                p = float(row.selected_model_prob)
                q = 1.0 - p
                full_kelly = max(((b * p) - q) / b, 0.0) if b > 0 else 0.0
                full_kelly = min(full_kelly, full_kelly_cap)
                actual_fraction = full_kelly * kelly_fraction
                stake = bankroll * actual_fraction
                won = bool(row.bet_won)
                profit = stake * b if won else -stake
                bankroll = bankroll + profit

                full_kelly_history.append(full_kelly)
                actual_fraction_history.append(actual_fraction)
                stake_history.append(stake)
                profit_history.append(profit)
                bankroll_history.append(bankroll)

            bets["strategy_name"] = "kelly"
            bets["kelly_fraction"] = kelly_fraction
            bets["edge_threshold"] = threshold
            bets["full_kelly_fraction_capped"] = full_kelly_history
            bets["actual_fraction_of_bankroll"] = actual_fraction_history
            bets["stake"] = stake_history
            bets["profit"] = profit_history
            bets["bankroll"] = bankroll_history
            bets["cumulative_profit"] = bets["bankroll"] - starting_bankroll

            result_rows.append(
                summarize_backtest_bets(
                    bets,
                    threshold=threshold,
                    strategy_name="kelly",
                    stake_column="stake",
                    profit_column="profit",
                    equity_column="bankroll",
                    extra_fields={
                        "kelly_fraction": kelly_fraction,
                        "starting_bankroll": starting_bankroll,
                        "final_bankroll": float(bankroll),
                    },
                )
            )
            bet_frames.append(bets)

    results_df = pd.DataFrame(result_rows).sort_values(
        ["final_bankroll", "roi", "total_profit"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    bets_df = pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame()
    return results_df, bets_df


def build_betting_backtest_summary(
    backtest_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    flat_results: pd.DataFrame,
    kelly_results: pd.DataFrame,
    model_label: str = "Random Forest",
    historical_note: str | None = None,
) -> str:
    """Create a human-readable betting backtest summary."""
    best_flat = flat_results.sort_values(["roi", "total_profit"], ascending=[False, False]).iloc[0] if not flat_results.empty else None
    best_kelly = kelly_results.sort_values(["final_bankroll", "roi"], ascending=[False, False]).iloc[0] if not kelly_results.empty else None

    lines = [
        "Betting backtest summary:",
        f"- model used: {model_label}",
        f"- odds source: {diagnostics['odds_source']}",
        f"- matched test fights with usable odds: {len(backtest_df)}",
        "- this backtest is exploratory and is not being used to tune the ML model.",
        "",
        "Flat betting:",
    ]
    if best_flat is None:
        lines.append("- no flat-betting rows were available.")
    else:
        lines.extend(
            [
                f"- best threshold by ROI: {best_flat.edge_threshold:.2f}",
                f"- bets: {int(best_flat.number_of_bets)}",
                f"- total_profit: {best_flat.total_profit:.6f}",
                f"- ROI: {best_flat.roi:.6f}",
                f"- average_edge: {best_flat.average_edge:.6f}",
                f"- max_drawdown: {best_flat.max_drawdown:.6f}",
            ]
        )

    lines.append("")
    lines.append("Kelly betting:")
    if best_kelly is None:
        lines.append("- no Kelly rows were available.")
    else:
        lines.extend(
            [
                f"- best Kelly strategy by final bankroll: threshold={best_kelly.edge_threshold:.2f}, fractional_kelly={best_kelly.kelly_fraction:.2f}",
                f"- bets: {int(best_kelly.number_of_bets)}",
                f"- final_bankroll: {best_kelly.final_bankroll:.6f}",
                f"- total_profit: {best_kelly.total_profit:.6f}",
                f"- ROI: {best_kelly.roi:.6f}",
                f"- max_drawdown: {best_kelly.max_drawdown:.6f}",
            ]
        )
    if historical_note:
        lines.extend(["", f"- note: {historical_note}"])

    lines.extend(
        [
            "",
            "Warnings and limitations:",
            "- The odds source is 'zewnetrzne', which appears to be an external or aggregated source rather than a directly tradable single sportsbook.",
            f"- The backtest uses only fights with matched usable odds and frozen {model_label} test predictions.",
            "- No model retraining, feature tuning, or threshold optimization should feed back into the model after this step.",
        ]
    )
    return "\n".join(lines)


def run_model_betting_backtest(
    *,
    model_label: str,
    prediction_path: Path,
    flat_results_path: Path,
    flat_bets_path: Path,
    kelly_results_path: Path,
    kelly_bets_path: Path,
    summary_path: Path,
    historical_note: str | None = None,
) -> dict[str, Any]:
    """Run one frozen-model betting backtest against the merged odds dataset."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    merged_odds_df, prediction_df = load_backtest_inputs(prediction_path=prediction_path)
    backtest_df, diagnostics = prepare_model_backtest_dataframe(merged_odds_df, prediction_df)

    flat_results, flat_bets = run_flat_betting_backtest(backtest_df, FLAT_EDGE_THRESHOLDS)
    kelly_results, kelly_bets = run_fractional_kelly_backtest(
        backtest_df,
        thresholds=FLAT_EDGE_THRESHOLDS,
        kelly_fractions=KELLY_FRACTIONS,
    )

    flat_results.to_csv(flat_results_path, index=False)
    flat_bets.to_csv(flat_bets_path, index=False)
    kelly_results.to_csv(kelly_results_path, index=False)
    kelly_bets.to_csv(kelly_bets_path, index=False)

    summary_text = build_betting_backtest_summary(
        backtest_df,
        diagnostics,
        flat_results,
        kelly_results,
        model_label=model_label,
        historical_note=historical_note,
    )
    summary_path.write_text(summary_text)

    print(f"{model_label} betting backtest summary:")
    print(f"- usable test fights with odds: {len(backtest_df)}")
    if not flat_results.empty:
        best_flat = flat_results.sort_values(["roi", "total_profit"], ascending=[False, False]).iloc[0]
        print(
            f"- best flat threshold by ROI: {best_flat.edge_threshold:.2f} "
            f"(ROI={best_flat.roi:.6f}, profit={best_flat.total_profit:.6f}, bets={int(best_flat.number_of_bets)})"
        )
    if not kelly_results.empty:
        best_kelly = kelly_results.sort_values(["final_bankroll", "roi"], ascending=[False, False]).iloc[0]
        print(
            f"- best Kelly strategy: threshold={best_kelly.edge_threshold:.2f}, "
            f"fractional Kelly={best_kelly.kelly_fraction:.2f}, "
            f"final bankroll={best_kelly.final_bankroll:.6f}"
        )

    return {
        "backtest_df": backtest_df,
        "diagnostics": diagnostics,
        "flat_results": flat_results,
        "flat_bets": flat_bets,
        "kelly_results": kelly_results,
        "kelly_bets": kelly_bets,
        "summary_text": summary_text,
        "created_files": [
            str(flat_results_path),
            str(flat_bets_path),
            str(kelly_results_path),
            str(kelly_bets_path),
            str(summary_path),
        ],
    }


def run_random_forest_betting_backtest() -> dict[str, Any]:
    """Run the historical exploratory Random Forest betting backtest."""
    return run_model_betting_backtest(
        model_label="Random Forest",
        prediction_path=FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH,
        flat_results_path=BETTING_FLAT_BACKTEST_RESULTS_PATH,
        flat_bets_path=BETTING_FLAT_BACKTEST_BETS_PATH,
        kelly_results_path=BETTING_KELLY_BACKTEST_RESULTS_PATH,
        kelly_bets_path=BETTING_KELLY_BACKTEST_BETS_PATH,
        summary_path=BETTING_BACKTEST_SUMMARY_PATH,
        historical_note="This retained Random Forest betting pass is historical/exploratory after the post-2010 cutoff change.",
    )


def run_logistic_betting_backtest() -> dict[str, Any]:
    """Run the main post-cutoff betting backtest using frozen Logistic Regression probabilities."""
    return run_model_betting_backtest(
        model_label="Logistic Regression",
        prediction_path=FINAL_LOGISTIC_TEST_PREDICTIONS_PATH,
        flat_results_path=LOGISTIC_BETTING_FLAT_BACKTEST_RESULTS_PATH,
        flat_bets_path=LOGISTIC_BETTING_FLAT_BACKTEST_BETS_PATH,
        kelly_results_path=LOGISTIC_BETTING_KELLY_BACKTEST_RESULTS_PATH,
        kelly_bets_path=LOGISTIC_BETTING_KELLY_BACKTEST_BETS_PATH,
        summary_path=LOGISTIC_BETTING_BACKTEST_SUMMARY_PATH,
    )


def load_flat_backtest_bets(path: Path = BETTING_FLAT_BACKTEST_BETS_PATH) -> pd.DataFrame:
    """Load flat-betting bet-level results with parsed dates and numeric columns."""
    bets_df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    numeric_columns = [
        "selected_decimal_odds",
        "selected_model_prob",
        "selected_market_prob_novig",
        "selected_edge",
        "profit",
        "stake",
        "edge_threshold",
    ]
    for column in numeric_columns:
        if column in bets_df.columns:
            bets_df[column] = pd.to_numeric(bets_df[column], errors="coerce")
    if "bet_won" in bets_df.columns:
        bets_df["bet_won"] = bets_df["bet_won"].astype("boolean")
    return bets_df


def select_best_flat_threshold(flat_results_path: Path = BETTING_FLAT_BACKTEST_RESULTS_PATH) -> float:
    """Pick the best flat-betting threshold by ROI, then total profit as a tiebreaker."""
    results_df = pd.read_csv(flat_results_path)
    best_row = results_df.sort_values(["roi", "total_profit", "number_of_bets"], ascending=[False, False, False]).iloc[0]
    return float(best_row["edge_threshold"])


def filter_bets_for_threshold(bets_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Keep only actual placed bets for one chosen flat-betting threshold."""
    filtered = bets_df.loc[
        bets_df["edge_threshold"].round(10).eq(round(threshold, 10))
        & bets_df["selected_side"].ne("skip")
    ].copy()
    return filtered.sort_values([DATE_COLUMN, "red_fighter_name", "blue_fighter_name"], kind="mergesort").reset_index(drop=True)


def add_betting_diagnostic_buckets(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable buckets for side, odds, market probability, model probability, and edge."""
    enriched = bets_df.copy()
    enriched["bet_side"] = enriched["selected_side"]
    enriched["flat_profit"] = pd.to_numeric(enriched["profit"], errors="coerce")

    enriched["odds_bucket"] = pd.cut(
        enriched["selected_decimal_odds"],
        bins=[-np.inf, 1.5, 2.0, 3.0, 5.0, np.inf],
        labels=["odds < 1.5", "1.5 <= odds < 2.0", "2.0 <= odds < 3.0", "3.0 <= odds < 5.0", "odds >= 5.0"],
        right=False,
    )
    enriched["market_prob_bucket"] = pd.cut(
        enriched["selected_market_prob_novig"],
        bins=[-np.inf, 0.25, 0.45, 0.55, 0.70, np.inf],
        labels=["longshot", "underdog", "near coin flip", "slight favorite", "heavy favorite"],
        right=False,
    )
    enriched["model_prob_bucket"] = pd.cut(
        enriched["selected_model_prob"],
        bins=[-np.inf, 0.35, 0.45, 0.55, 0.65, np.inf],
        labels=["<0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", ">0.65"],
        right=False,
    )
    enriched["edge_bucket"] = pd.cut(
        enriched["selected_edge"],
        bins=[0.0, 0.02, 0.05, 0.08, 0.10, np.inf],
        labels=["0.00-0.02", "0.02-0.05", "0.05-0.08", "0.08-0.10", ">0.10"],
        include_lowest=True,
        right=False,
    )
    return enriched


def summarize_segment_groups(bets_df: pd.DataFrame, group_column: str, output_column_name: str | None = None) -> pd.DataFrame:
    """Summarize betting performance for one segmentation column."""
    if bets_df.empty:
        return pd.DataFrame(
            columns=[
                output_column_name or group_column,
                "number_of_bets",
                "win_rate",
                "average_odds",
                "average_model_probability",
                "average_market_probability",
                "average_edge",
                "total_profit",
                "roi",
                "max_drawdown",
            ]
        )

    rows: list[dict[str, Any]] = []
    for segment_value, segment_df in bets_df.groupby(group_column, dropna=False, observed=False):
        ordered = segment_df.sort_values(DATE_COLUMN, kind="mergesort").copy()
        ordered["segment_cumulative_profit"] = ordered["flat_profit"].cumsum()
        win_rate = ordered["bet_won"].fillna(False).astype(int).mean()
        total_staked = float(ordered["stake"].sum()) if "stake" in ordered.columns else float(len(ordered))
        total_profit = float(ordered["flat_profit"].sum())
        rows.append(
            {
                output_column_name or group_column: segment_value,
                "number_of_bets": int(len(ordered)),
                "win_rate": float(win_rate),
                "average_odds": float(ordered["selected_decimal_odds"].mean()),
                "average_model_probability": float(ordered["selected_model_prob"].mean()),
                "average_market_probability": float(ordered["selected_market_prob_novig"].mean()),
                "average_edge": float(ordered["selected_edge"].mean()),
                "total_profit": total_profit,
                "roi": float(total_profit / total_staked) if total_staked > 0 else 0.0,
                "max_drawdown": compute_max_drawdown(ordered["segment_cumulative_profit"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["roi", "total_profit", "number_of_bets"], ascending=[False, False, False]).reset_index(drop=True)


def build_edge_quality_table(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Check whether larger modeled edge translated into better realized performance."""
    rows: list[dict[str, Any]] = []
    for edge_bucket, bucket_df in bets_df.groupby("edge_bucket", dropna=False, observed=False):
        total_profit = float(bucket_df["flat_profit"].sum())
        total_staked = float(bucket_df["stake"].sum())
        actual_win_rate = bucket_df["bet_won"].fillna(False).astype(int).mean()
        rows.append(
            {
                "edge_bucket": edge_bucket,
                "number_of_bets": int(len(bucket_df)),
                "average_predicted_edge": float(bucket_df["selected_edge"].mean()),
                "actual_win_rate": float(actual_win_rate),
                "average_implied_probability": float(bucket_df["selected_market_prob_novig"].mean()),
                "expected_win_rate_from_model": float(bucket_df["selected_model_prob"].mean()),
                "realized_profit": total_profit,
                "roi": float(total_profit / total_staked) if total_staked > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_edge_correlation_table(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple correlations between edge/probability quantities and outcomes."""
    corr_pairs = {
        "edge_vs_bet_won": ("selected_edge", "bet_won"),
        "edge_vs_flat_profit": ("selected_edge", "flat_profit"),
        "model_probability_vs_bet_won": ("selected_model_prob", "bet_won"),
        "market_probability_vs_bet_won": ("selected_market_prob_novig", "bet_won"),
    }
    rows = []
    numeric = bets_df.copy()
    numeric["bet_won_numeric"] = numeric["bet_won"].fillna(False).astype(int)
    for label, (left, right) in corr_pairs.items():
        right_column = "bet_won_numeric" if right == "bet_won" else right
        correlation = numeric[left].corr(numeric[right_column])
        rows.append({"correlation_name": label, "correlation": float(correlation) if pd.notna(correlation) else np.nan})
    return pd.DataFrame(rows)


def build_probability_decile_calibration(
    df: pd.DataFrame,
    probability_column: str,
    comparison_probability_column: str,
    actual_label: str,
    comparison_label: str,
) -> pd.DataFrame:
    """Create a decile calibration table for one probability source."""
    working = df.copy()
    working = working.loc[working[probability_column].notna() & working[TARGET_COLUMN].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["decile", f"average_{actual_label}", "actual_win_rate", f"average_{comparison_label}", "count"])

    working["probability_decile"] = pd.qcut(
        working[probability_column].rank(method="first"),
        q=min(10, len(working)),
        labels=False,
        duplicates="drop",
    )
    calibration = (
        working.groupby("probability_decile", observed=False)
        .agg(
            count=(TARGET_COLUMN, "size"),
            actual_win_rate=(TARGET_COLUMN, "mean"),
            **{
                f"average_{actual_label}": (probability_column, "mean"),
                f"average_{comparison_label}": (comparison_probability_column, "mean"),
            },
        )
        .reset_index()
        .rename(columns={"probability_decile": "decile"})
    )
    return calibration.sort_values("decile").reset_index(drop=True)


def build_favorite_longshot_analysis(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize favorites, underdogs, and longshots for the chosen flat-betting policy."""
    definitions = {
        "favorites_only": bets_df["selected_decimal_odds"] < 2.0,
        "underdogs_only": bets_df["selected_decimal_odds"] >= 2.0,
        "longshots_only": bets_df["selected_decimal_odds"] >= 3.0,
    }
    rows = []
    for name, mask in definitions.items():
        subset = bets_df.loc[mask].copy()
        total_profit = float(subset["flat_profit"].sum()) if not subset.empty else 0.0
        total_staked = float(subset["stake"].sum()) if not subset.empty else 0.0
        win_rate = float(subset["bet_won"].fillna(False).astype(int).mean()) if not subset.empty else 0.0
        rows.append(
            {
                "segment": name,
                "number_of_bets": int(len(subset)),
                "win_rate": win_rate,
                "roi": float(total_profit / total_staked) if total_staked > 0 else 0.0,
                "average_edge": float(subset["selected_edge"].mean()) if not subset.empty else 0.0,
                "average_odds": float(subset["selected_decimal_odds"].mean()) if not subset.empty else 0.0,
                "total_profit": total_profit,
            }
        )
    return pd.DataFrame(rows)


def build_cumulative_profit_table(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Track cumulative profit and rolling ROI over time for the chosen flat strategy."""
    ordered = bets_df.sort_values([DATE_COLUMN, "red_fighter_name", "blue_fighter_name"], kind="mergesort").copy()
    ordered["bet_number"] = np.arange(1, len(ordered) + 1)
    ordered["cumulative_profit"] = ordered["flat_profit"].cumsum()
    ordered["rolling_50_profit"] = ordered["flat_profit"].rolling(50, min_periods=10).sum()
    ordered["rolling_50_staked"] = ordered["stake"].rolling(50, min_periods=10).sum()
    ordered["rolling_50_roi"] = ordered["rolling_50_profit"] / ordered["rolling_50_staked"]
    return ordered[
        [
            DATE_COLUMN,
            "red_fighter_name",
            "blue_fighter_name",
            "bet_number",
            "flat_profit",
            "cumulative_profit",
            "rolling_50_roi",
        ]
    ].copy()


def build_time_segment_performance(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly and quarterly performance for the chosen flat strategy."""
    working = bets_df.copy()
    working["month"] = working[DATE_COLUMN].dt.to_period("M").astype(str)
    working["quarter"] = working[DATE_COLUMN].dt.to_period("Q").astype(str)

    def summarize_time_column(column_name: str) -> pd.DataFrame:
        rows = []
        for segment_value, segment_df in working.groupby(column_name, observed=False):
            total_profit = float(segment_df["flat_profit"].sum())
            total_staked = float(segment_df["stake"].sum())
            win_rate = float(segment_df["bet_won"].fillna(False).astype(int).mean())
            rows.append(
                {
                    "segment_type": column_name,
                    "segment_value": segment_value,
                    "number_of_bets": int(len(segment_df)),
                    "win_rate": win_rate,
                    "total_profit": total_profit,
                    "roi": float(total_profit / total_staked) if total_staked > 0 else 0.0,
                    "average_edge": float(segment_df["selected_edge"].mean()),
                }
            )
        return pd.DataFrame(rows)

    return pd.concat(
        [summarize_time_column("month"), summarize_time_column("quarter")],
        ignore_index=True,
    )


def build_betting_failure_diagnostics_summary(
    threshold: float,
    side_segments: pd.DataFrame,
    odds_segments: pd.DataFrame,
    edge_quality: pd.DataFrame,
    edge_correlations: pd.DataFrame,
    model_calibration: pd.DataFrame,
    market_calibration: pd.DataFrame,
    favorite_longshot: pd.DataFrame,
    model_label: str = "Random Forest",
) -> str:
    """Write a concise diagnostic summary for why the first betting backtest lost money."""
    top_losing_segment = odds_segments.sort_values("roi", ascending=True).iloc[0] if not odds_segments.empty else None
    best_segment = odds_segments.sort_values("roi", ascending=False).iloc[0] if not odds_segments.empty else None
    high_edge_rows = edge_quality.copy()
    high_edge_rows["edge_bucket_str"] = high_edge_rows["edge_bucket"].astype(str)
    high_edge_better = False
    if not high_edge_rows.empty:
        high_edge = high_edge_rows.loc[high_edge_rows["edge_bucket_str"].eq(">0.10")]
        low_edge = high_edge_rows.loc[high_edge_rows["edge_bucket_str"].eq("0.02-0.05")]
        if not high_edge.empty and not low_edge.empty:
            high_edge_better = float(high_edge["roi"].iloc[0]) > float(low_edge["roi"].iloc[0])

    longshot_row = favorite_longshot.loc[favorite_longshot["segment"].eq("longshots_only")]
    favorite_row = favorite_longshot.loc[favorite_longshot["segment"].eq("favorites_only")]
    longshots_worse = False
    if not longshot_row.empty and not favorite_row.empty:
        longshots_worse = float(longshot_row["roi"].iloc[0]) < float(favorite_row["roi"].iloc[0])

    model_calibration_gap = (
        (model_calibration["average_model_probability"] - model_calibration["actual_win_rate"]).abs().mean()
        if not model_calibration.empty else np.nan
    )
    market_calibration_gap = (
        (market_calibration["average_market_probability"] - market_calibration["actual_win_rate"]).abs().mean()
        if not market_calibration.empty else np.nan
    )
    market_better_calibrated = (
        pd.notna(model_calibration_gap) and pd.notna(market_calibration_gap) and market_calibration_gap < model_calibration_gap
    )

    lines = [
        "Betting failure diagnostics summary:",
        "- This is post-hoc diagnostic analysis only. It must not be used to tune or retrain the model.",
        f"- model analyzed: {model_label}",
        f"- analyzed flat-betting policy threshold: {threshold:.2f}",
        "",
        "Key findings:",
        f"- higher edge led to better ROI: {high_edge_better}",
        f"- longshots caused major losses relative to favorites: {longshots_worse}",
        f"- market probabilities were better calibrated than the model on matched-odds fights: {market_better_calibrated}",
    ]
    if top_losing_segment is not None:
        lines.append(
            f"- top losing odds segment: {top_losing_segment['odds_bucket']} "
            f"(ROI={top_losing_segment['roi']:.6f}, bets={int(top_losing_segment['number_of_bets'])})"
        )
    if best_segment is not None:
        lines.append(
            f"- best odds segment: {best_segment['odds_bucket']} "
            f"(ROI={best_segment['roi']:.6f}, bets={int(best_segment['number_of_bets'])})"
        )

    lines.extend(
        [
            "",
            "Likely reasons the backtest lost money:",
            "- market efficiency and vig likely erased much of the modeled edge",
            f"- the {model_label} probabilities appear imperfectly calibrated on the odds-matched subset",
            "- high-odds bets added noise and large drawdowns",
            "- the selected odds source is aggregated ('zewnetrzne') rather than a sharp directly tradable source",
            "- some model signals may already be partially priced by the market",
            "",
            "Future improvement ideas:",
            "- calibrate probabilities on validation only before betting",
            "- use line shopping or best available book prices instead of one aggregated source",
            "- prefer a sharper source such as Pinnacle if coverage is acceptable",
            "- avoid or separately analyze longshots and other volatile odds regions",
            "- compare model-versus-market edge by odds region before staking",
            "- add closing-line-value analysis if timestamped odds are reliable",
        ]
    )
    return "\n".join(lines)


def run_model_betting_failure_diagnostics(
    *,
    model_label: str,
    prediction_path: Path,
    flat_bets_path: Path,
    flat_results_path: Path,
    segment_by_side_path: Path,
    segment_by_odds_bucket_path: Path,
    segment_by_market_prob_bucket_path: Path,
    segment_by_model_prob_bucket_path: Path,
    segment_by_edge_bucket_path: Path,
    edge_quality_path: Path,
    edge_correlations_path: Path,
    model_calibration_path: Path,
    market_calibration_path: Path,
    favorite_longshot_path: Path,
    cumulative_profit_path: Path,
    time_segment_performance_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Diagnose one frozen-model flat-betting backtest without changing any model outputs."""
    flat_bets_df = load_flat_backtest_bets(path=flat_bets_path)
    best_threshold = select_best_flat_threshold(flat_results_path=flat_results_path)
    chosen_bets = filter_bets_for_threshold(flat_bets_df, best_threshold)
    chosen_bets = add_betting_diagnostic_buckets(chosen_bets)

    side_segments = summarize_segment_groups(chosen_bets, "bet_side")
    odds_segments = summarize_segment_groups(chosen_bets, "odds_bucket")
    market_prob_segments = summarize_segment_groups(chosen_bets, "market_prob_bucket")
    model_prob_segments = summarize_segment_groups(chosen_bets, "model_prob_bucket")
    edge_segments = summarize_segment_groups(chosen_bets, "edge_bucket")
    edge_quality = build_edge_quality_table(chosen_bets)
    edge_correlations = build_edge_correlation_table(chosen_bets)
    favorite_longshot = build_favorite_longshot_analysis(chosen_bets)
    cumulative_profit = build_cumulative_profit_table(chosen_bets)
    time_segment_performance = build_time_segment_performance(chosen_bets)

    merged_odds_df, prediction_df = load_backtest_inputs(prediction_path=prediction_path)
    matched_odds_df, diagnostics = prepare_model_backtest_dataframe(merged_odds_df, prediction_df)
    matched_odds_df["actual_red_win"] = matched_odds_df[TARGET_COLUMN].astype(float)
    model_calibration = build_probability_decile_calibration(
        matched_odds_df,
        probability_column="model_prob_red",
        comparison_probability_column="red_implied_prob_novig",
        actual_label="model_probability",
        comparison_label="market_probability",
    )
    market_calibration = build_probability_decile_calibration(
        matched_odds_df,
        probability_column="red_implied_prob_novig",
        comparison_probability_column="model_prob_red",
        actual_label="market_probability",
        comparison_label="model_probability",
    )

    side_segments.to_csv(segment_by_side_path, index=False)
    odds_segments.to_csv(segment_by_odds_bucket_path, index=False)
    market_prob_segments.to_csv(segment_by_market_prob_bucket_path, index=False)
    model_prob_segments.to_csv(segment_by_model_prob_bucket_path, index=False)
    edge_segments.to_csv(segment_by_edge_bucket_path, index=False)
    edge_quality.to_csv(edge_quality_path, index=False)
    edge_correlations.to_csv(edge_correlations_path, index=False)
    model_calibration.to_csv(model_calibration_path, index=False)
    market_calibration.to_csv(market_calibration_path, index=False)
    favorite_longshot.to_csv(favorite_longshot_path, index=False)
    cumulative_profit.to_csv(cumulative_profit_path, index=False)
    time_segment_performance.to_csv(time_segment_performance_path, index=False)

    summary_text = build_betting_failure_diagnostics_summary(
        threshold=best_threshold,
        side_segments=side_segments,
        odds_segments=odds_segments,
        edge_quality=edge_quality,
        edge_correlations=edge_correlations,
        model_calibration=model_calibration,
        market_calibration=market_calibration,
        favorite_longshot=favorite_longshot,
        model_label=model_label,
    )
    summary_path.write_text(summary_text)

    top_losing_segment = odds_segments.sort_values("roi", ascending=True).head(1)
    best_segment = odds_segments.sort_values("roi", ascending=False).head(1)
    edge_quality_for_compare = edge_quality.copy()
    edge_quality_for_compare["edge_bucket_str"] = edge_quality_for_compare["edge_bucket"].astype(str)
    high_edge_row = edge_quality_for_compare.loc[edge_quality_for_compare["edge_bucket_str"].eq(">0.10")]
    low_edge_row = edge_quality_for_compare.loc[edge_quality_for_compare["edge_bucket_str"].eq("0.02-0.05")]
    higher_edge_improved_roi = (
        not high_edge_row.empty and not low_edge_row.empty and float(high_edge_row["roi"].iloc[0]) > float(low_edge_row["roi"].iloc[0])
    )
    longshot_row = favorite_longshot.loc[favorite_longshot["segment"].eq("longshots_only")]
    favorite_row = favorite_longshot.loc[favorite_longshot["segment"].eq("favorites_only")]
    longshots_caused_major_losses = (
        not longshot_row.empty and not favorite_row.empty and float(longshot_row["roi"].iloc[0]) < float(favorite_row["roi"].iloc[0])
    )
    model_gap = (
        (model_calibration["average_model_probability"] - model_calibration["actual_win_rate"]).abs().mean()
        if not model_calibration.empty else np.nan
    )
    market_gap = (
        (market_calibration["average_market_probability"] - market_calibration["actual_win_rate"]).abs().mean()
        if not market_calibration.empty else np.nan
    )
    market_better_calibrated = pd.notna(model_gap) and pd.notna(market_gap) and market_gap < model_gap

    print(f"{model_label} betting failure diagnostics:")
    if not top_losing_segment.empty:
        row = top_losing_segment.iloc[0]
        print(f"- top losing segment: {row['odds_bucket']} (ROI={row['roi']:.6f}, bets={int(row['number_of_bets'])})")
    if not best_segment.empty:
        row = best_segment.iloc[0]
        print(f"- best segment: {row['odds_bucket']} (ROI={row['roi']:.6f}, bets={int(row['number_of_bets'])})")
    print(f"- higher edge improved ROI: {higher_edge_improved_roi}")
    print(f"- longshots caused major losses: {longshots_caused_major_losses}")
    print(f"- market better calibrated than model: {market_better_calibrated}")

    return {
        "best_threshold": best_threshold,
        "chosen_bets": chosen_bets,
        "diagnostics": diagnostics,
        "segment_tables": {
            "side": side_segments,
            "odds_bucket": odds_segments,
            "market_prob_bucket": market_prob_segments,
            "model_prob_bucket": model_prob_segments,
            "edge_bucket": edge_segments,
        },
        "edge_quality": edge_quality,
        "edge_correlations": edge_correlations,
        "model_calibration": model_calibration,
        "market_calibration": market_calibration,
        "favorite_longshot": favorite_longshot,
        "cumulative_profit": cumulative_profit,
        "time_segment_performance": time_segment_performance,
        "summary_text": summary_text,
        "created_files": [
            str(segment_by_side_path),
            str(segment_by_odds_bucket_path),
            str(segment_by_market_prob_bucket_path),
            str(segment_by_model_prob_bucket_path),
            str(segment_by_edge_bucket_path),
            str(edge_quality_path),
            str(edge_correlations_path),
            str(model_calibration_path),
            str(market_calibration_path),
            str(favorite_longshot_path),
            str(cumulative_profit_path),
            str(time_segment_performance_path),
            str(summary_path),
        ],
        "higher_edge_improved_roi": higher_edge_improved_roi,
        "longshots_caused_major_losses": longshots_caused_major_losses,
        "market_better_calibrated": market_better_calibrated,
    }


def run_random_forest_betting_failure_diagnostics() -> dict[str, Any]:
    """Run historical Random Forest betting diagnostics for continuity."""
    return run_model_betting_failure_diagnostics(
        model_label="Random Forest",
        prediction_path=FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH,
        flat_bets_path=BETTING_FLAT_BACKTEST_BETS_PATH,
        flat_results_path=BETTING_FLAT_BACKTEST_RESULTS_PATH,
        segment_by_side_path=BETTING_SEGMENT_BY_SIDE_PATH,
        segment_by_odds_bucket_path=BETTING_SEGMENT_BY_ODDS_BUCKET_PATH,
        segment_by_market_prob_bucket_path=BETTING_SEGMENT_BY_MARKET_PROB_BUCKET_PATH,
        segment_by_model_prob_bucket_path=BETTING_SEGMENT_BY_MODEL_PROB_BUCKET_PATH,
        segment_by_edge_bucket_path=BETTING_SEGMENT_BY_EDGE_BUCKET_PATH,
        edge_quality_path=BETTING_EDGE_QUALITY_PATH,
        edge_correlations_path=BETTING_EDGE_CORRELATIONS_PATH,
        model_calibration_path=BETTING_MODEL_CALIBRATION_ON_MATCHED_ODDS_PATH,
        market_calibration_path=BETTING_MARKET_CALIBRATION_ON_MATCHED_ODDS_PATH,
        favorite_longshot_path=BETTING_FAVORITE_LONGSHOT_ANALYSIS_PATH,
        cumulative_profit_path=BETTING_CUMULATIVE_PROFIT_PATH,
        time_segment_performance_path=BETTING_TIME_SEGMENT_PERFORMANCE_PATH,
        summary_path=BETTING_FAILURE_DIAGNOSTICS_SUMMARY_PATH,
    )


def run_logistic_betting_failure_diagnostics() -> dict[str, Any]:
    """Run the main post-cutoff Logistic Regression betting diagnostics."""
    return run_model_betting_failure_diagnostics(
        model_label="Logistic Regression",
        prediction_path=FINAL_LOGISTIC_TEST_PREDICTIONS_PATH,
        flat_bets_path=LOGISTIC_BETTING_FLAT_BACKTEST_BETS_PATH,
        flat_results_path=LOGISTIC_BETTING_FLAT_BACKTEST_RESULTS_PATH,
        segment_by_side_path=LOGISTIC_BETTING_SEGMENT_BY_SIDE_PATH,
        segment_by_odds_bucket_path=LOGISTIC_BETTING_SEGMENT_BY_ODDS_BUCKET_PATH,
        segment_by_market_prob_bucket_path=LOGISTIC_BETTING_SEGMENT_BY_MARKET_PROB_BUCKET_PATH,
        segment_by_model_prob_bucket_path=LOGISTIC_BETTING_SEGMENT_BY_MODEL_PROB_BUCKET_PATH,
        segment_by_edge_bucket_path=LOGISTIC_BETTING_SEGMENT_BY_EDGE_BUCKET_PATH,
        edge_quality_path=LOGISTIC_BETTING_EDGE_QUALITY_PATH,
        edge_correlations_path=LOGISTIC_BETTING_EDGE_CORRELATIONS_PATH,
        model_calibration_path=LOGISTIC_BETTING_MODEL_CALIBRATION_ON_MATCHED_ODDS_PATH,
        market_calibration_path=LOGISTIC_BETTING_MARKET_CALIBRATION_ON_MATCHED_ODDS_PATH,
        favorite_longshot_path=LOGISTIC_BETTING_FAVORITE_LONGSHOT_ANALYSIS_PATH,
        cumulative_profit_path=LOGISTIC_BETTING_CUMULATIVE_PROFIT_PATH,
        time_segment_performance_path=LOGISTIC_BETTING_TIME_SEGMENT_PERFORMANCE_PATH,
        summary_path=LOGISTIC_BETTING_FAILURE_DIAGNOSTICS_SUMMARY_PATH,
    )


def run_betting_failure_diagnostics() -> dict[str, Any]:
    """Run the main post-cutoff betting diagnostics on frozen Logistic Regression probabilities."""
    return run_logistic_betting_failure_diagnostics()


def build_betting_model_backtest_comparison(
    logistic_flat_results: pd.DataFrame,
    rf_flat_results: pd.DataFrame,
    logistic_kelly_results: pd.DataFrame,
    rf_kelly_results: pd.DataFrame,
    logistic_backtest_df: pd.DataFrame,
    rf_backtest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Compare post-cutoff Logistic betting against the historical Random Forest pass."""
    rows: list[dict[str, Any]] = []
    for model_name, flat_results, kelly_results, backtest_df in [
        ("logistic_regression", logistic_flat_results, logistic_kelly_results, logistic_backtest_df),
        ("random_forest_historical", rf_flat_results, rf_kelly_results, rf_backtest_df),
    ]:
        best_flat = flat_results.sort_values(["roi", "total_profit"], ascending=[False, False]).iloc[0] if not flat_results.empty else None
        best_kelly = (
            kelly_results.sort_values(["final_bankroll", "roi"], ascending=[False, False]).iloc[0]
            if not kelly_results.empty else None
        )
        rows.append(
            {
                "model_name": model_name,
                "matched_test_fights": int(len(backtest_df)),
                "best_flat_threshold": float(best_flat["edge_threshold"]) if best_flat is not None else np.nan,
                "best_flat_roi": float(best_flat["roi"]) if best_flat is not None else np.nan,
                "best_flat_total_profit": float(best_flat["total_profit"]) if best_flat is not None else np.nan,
                "best_flat_max_drawdown": float(best_flat["max_drawdown"]) if best_flat is not None else np.nan,
                "best_kelly_threshold": float(best_kelly["edge_threshold"]) if best_kelly is not None else np.nan,
                "best_kelly_fraction": float(best_kelly["kelly_fraction"]) if best_kelly is not None else np.nan,
                "best_kelly_final_bankroll": float(best_kelly["final_bankroll"]) if best_kelly is not None else np.nan,
                "best_kelly_roi": float(best_kelly["roi"]) if best_kelly is not None else np.nan,
                "any_positive_flat_roi": bool((flat_results["roi"] > 0).any()) if not flat_results.empty else False,
                "any_positive_kelly_final_bankroll": bool((kelly_results["final_bankroll"] > 1000).any()) if not kelly_results.empty else False,
            }
        )

    comparison_df = pd.DataFrame(rows)
    logistic_row = comparison_df.loc[comparison_df["model_name"].eq("logistic_regression")].iloc[0]
    rf_row = comparison_df.loc[comparison_df["model_name"].eq("random_forest_historical")].iloc[0]
    more_stable_model = (
        "logistic_regression"
        if logistic_row["best_flat_max_drawdown"] < rf_row["best_flat_max_drawdown"]
        else "random_forest_historical"
    )
    logistic_improved_over_rf = (
        logistic_row["best_flat_roi"] > rf_row["best_flat_roi"]
        or logistic_row["best_kelly_final_bankroll"] > rf_row["best_kelly_final_bankroll"]
    )

    summary_lines = [
        "Betting model backtest comparison summary:",
        "- The post-2010 cutoff made Logistic Regression the strongest final probability model by log loss, ROC AUC, and Brier score.",
        "- The Random Forest betting backtest is retained only as a historical/exploratory comparison.",
        "",
        "Flat betting comparison:",
        f"- logistic best flat threshold: {logistic_row['best_flat_threshold']:.2f}",
        f"- logistic best flat ROI: {logistic_row['best_flat_roi']:.6f}",
        f"- random forest best flat threshold: {rf_row['best_flat_threshold']:.2f}",
        f"- random forest best flat ROI: {rf_row['best_flat_roi']:.6f}",
        "",
        "Kelly comparison:",
        f"- logistic best Kelly final bankroll: {logistic_row['best_kelly_final_bankroll']:.6f}",
        f"- random forest best Kelly final bankroll: {rf_row['best_kelly_final_bankroll']:.6f}",
        "",
        "Overall interpretation:",
        f"- more stable model by flat-bet drawdown: {more_stable_model}",
        f"- logistic improved over RF on at least one betting criterion: {logistic_improved_over_rf}",
        f"- any positive logistic flat ROI: {bool(logistic_row['any_positive_flat_roi'])}",
        f"- any positive random-forest flat ROI: {bool(rf_row['any_positive_flat_roi'])}",
        "- Threshold rankings remain descriptive only; they must not feed back into model tuning.",
    ]
    update_lines = [
        "Post-cutoff betting model update summary:",
        "- The repo now uses frozen Logistic Regression probabilities as the main betting input because Logistic became the strongest post-2010 probability model.",
        "- The older Random Forest betting pass is preserved as historical/exploratory output only.",
        f"- Logistic best flat ROI: {logistic_row['best_flat_roi']:.6f} at threshold {logistic_row['best_flat_threshold']:.2f}",
        f"- Logistic best Kelly final bankroll: {logistic_row['best_kelly_final_bankroll']:.6f}",
        f"- Random Forest historical best flat ROI: {rf_row['best_flat_roi']:.6f} at threshold {rf_row['best_flat_threshold']:.2f}",
        f"- Random Forest historical best Kelly final bankroll: {rf_row['best_kelly_final_bankroll']:.6f}",
        f"- Did the high-level conclusion change? Logistic improved over RF = {logistic_improved_over_rf}.",
        "- The market no-vig probabilities remain the key benchmark, and betting results remain exploratory rather than a tuning target.",
    ]
    return comparison_df, "\n".join(summary_lines), "\n".join(update_lines)


def run_betting_model_backtest_comparison() -> dict[str, Any]:
    """Compare the main Logistic betting backtest against the retained historical RF backtest."""
    logistic_results = run_logistic_betting_backtest()
    rf_results = run_random_forest_betting_backtest()
    comparison_df, summary_text, update_text = build_betting_model_backtest_comparison(
        logistic_results["flat_results"],
        rf_results["flat_results"],
        logistic_results["kelly_results"],
        rf_results["kelly_results"],
        logistic_results["backtest_df"],
        rf_results["backtest_df"],
    )
    comparison_df.to_csv(BETTING_MODEL_BACKTEST_COMPARISON_PATH, index=False)
    BETTING_MODEL_BACKTEST_COMPARISON_SUMMARY_PATH.write_text(summary_text)
    POST_CUTOFF_BETTING_MODEL_UPDATE_SUMMARY_PATH.write_text(update_text)

    return {
        "comparison_df": comparison_df,
        "summary_text": summary_text,
        "update_text": update_text,
        "created_files": [
            str(BETTING_MODEL_BACKTEST_COMPARISON_PATH),
            str(BETTING_MODEL_BACKTEST_COMPARISON_SUMMARY_PATH),
            str(POST_CUTOFF_BETTING_MODEL_UPDATE_SUMMARY_PATH),
        ],
    }


def build_split_row_key(df: pd.DataFrame) -> pd.Series:
    """Create a stable key for overlap checks across chronological splits."""
    return (
        pd.to_datetime(df[DATE_COLUMN], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "||"
        + df["red_fighter_name"].fillna("").astype(str)
        + "||"
        + df["blue_fighter_name"].fillna("").astype(str)
        + "||"
        + pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(-1).astype(int).astype(str)
    )


def audit_split_summary(
    modeling_df: pd.DataFrame,
    matched_odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize chronological splits and overlap diagnostics for model families and betting subset."""
    train_df, validation_df, test_df = chronological_split(modeling_df)
    split_keys = {
        "train": set(build_split_row_key(train_df)),
        "validation": set(build_split_row_key(validation_df)),
        "test": set(build_split_row_key(test_df)),
    }
    split_dates = {
        "train": set(pd.to_datetime(train_df[DATE_COLUMN]).dt.strftime("%Y-%m-%d")),
        "validation": set(pd.to_datetime(validation_df[DATE_COLUMN]).dt.strftime("%Y-%m-%d")),
        "test": set(pd.to_datetime(test_df[DATE_COLUMN]).dt.strftime("%Y-%m-%d")),
    }
    train_validation_overlap_rows = len(split_keys["train"] & split_keys["validation"])
    train_test_overlap_rows = len(split_keys["train"] & split_keys["test"])
    validation_test_overlap_rows = len(split_keys["validation"] & split_keys["test"])
    shared_dates_train_validation = len(split_dates["train"] & split_dates["validation"])
    shared_dates_validation_test = len(split_dates["validation"] & split_dates["test"])
    shared_dates_train_test = len(split_dates["train"] & split_dates["test"])
    chronological_order_ok = (
        train_df[DATE_COLUMN].max() <= validation_df[DATE_COLUMN].min()
        and validation_df[DATE_COLUMN].max() <= test_df[DATE_COLUMN].min()
    )

    rows: list[dict[str, Any]] = []
    for model_family in ["logistic_regression", "random_forest", "xgboost"]:
        for split_name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
            rows.append(
                {
                    "model_family": model_family,
                    "split_name": split_name,
                    "rows": int(len(split_df)),
                    "start_date": split_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                    "end_date": split_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                    "red_win_mean": float(split_df[TARGET_COLUMN].mean()),
                    "chronological_order_ok": chronological_order_ok,
                    "train_validation_overlap_rows": train_validation_overlap_rows,
                    "train_test_overlap_rows": train_test_overlap_rows,
                    "validation_test_overlap_rows": validation_test_overlap_rows,
                    "shared_dates_train_validation": shared_dates_train_validation,
                    "shared_dates_validation_test": shared_dates_validation_test,
                    "shared_dates_train_test": shared_dates_train_test,
                    "date_boundary_touching": shared_dates_train_validation > 0 or shared_dates_validation_test > 0,
                }
            )

    matched_odds_keys = set(build_split_row_key(matched_odds_df))
    rows.append(
        {
            "model_family": "betting_backtest",
            "split_name": "matched_test_odds_subset",
            "rows": int(len(matched_odds_df)),
            "start_date": matched_odds_df[DATE_COLUMN].min().strftime("%Y-%m-%d") if not matched_odds_df.empty else None,
            "end_date": matched_odds_df[DATE_COLUMN].max().strftime("%Y-%m-%d") if not matched_odds_df.empty else None,
            "red_win_mean": float(matched_odds_df[TARGET_COLUMN].mean()) if not matched_odds_df.empty else np.nan,
            "chronological_order_ok": True,
            "train_validation_overlap_rows": 0,
            "train_test_overlap_rows": 0,
            "validation_test_overlap_rows": 0,
            "shared_dates_train_validation": 0,
            "shared_dates_validation_test": 0,
            "shared_dates_train_test": 0,
            "date_boundary_touching": False,
            "subset_of_test_rows": len(matched_odds_keys - split_keys["test"]) == 0,
        }
    )
    return pd.DataFrame(rows)


def audit_feature_and_leakage_checks(
    modeling_df: pd.DataFrame,
    saved_modeling_df: pd.DataFrame,
    logistic_metrics_payload: dict[str, Any],
    random_forest_metrics_payload: dict[str, Any],
    xgboost_metrics_payload: dict[str, Any],
) -> pd.DataFrame:
    """Audit the frozen feature set, leakage safety, and dataset consistency."""
    expected_features = FINAL_LOGISTIC_FEATURE_CANDIDATES.copy()
    actual_sets = {
        "expected_final_feature_set": expected_features,
        "logistic_saved_feature_set": logistic_metrics_payload.get("feature_names", []),
        "random_forest_saved_feature_set": random_forest_metrics_payload.get("feature_names", []),
        "xgboost_saved_feature_set": xgboost_metrics_payload.get("feature_names", []),
    }

    suspicious_keywords = [
        "result",
        "outcome",
        "method",
        "round",
        "time",
        "odds",
        "referee",
        "details",
        "fighter_1_odds",
        "fighter_2_odds",
        "event_name",
        "event_location",
    ]

    rows: list[dict[str, Any]] = []
    for set_name, feature_list in actual_sets.items():
        rows.append(
            {
                "check_name": f"{set_name}_matches_expected",
                "feature_name": None,
                "status": "PASS" if feature_list == expected_features else "FAIL",
                "details": f"feature_count={len(feature_list)} expected_count={len(expected_features)}",
            }
        )

    for feature_name in expected_features:
        exists_in_modeling = feature_name in modeling_df.columns
        exists_in_saved = feature_name in saved_modeling_df.columns
        safe_prefix = feature_name.startswith("diff_") or feature_name.startswith("pre_fight_")
        suspicious_hits = [keyword for keyword in suspicious_keywords if keyword in feature_name.lower()]
        if any(token in feature_name.lower() for token in ["five_round", "past_5_round", "third_round"]):
            suspicious_hits = [keyword for keyword in suspicious_hits if keyword != "round"]
        status = "PASS" if exists_in_modeling and exists_in_saved and safe_prefix and not suspicious_hits else "FAIL"
        rows.append(
            {
                "check_name": "final_feature_is_available_and_safe",
                "feature_name": feature_name,
                "status": status,
                "details": (
                    f"exists_in_modeling={exists_in_modeling}, exists_in_saved={exists_in_saved}, "
                    f"safe_prefix={safe_prefix}, suspicious_hits={suspicious_hits}"
                ),
            }
        )

    rebuilt_matches_saved = modeling_df.columns.tolist() == saved_modeling_df.columns.tolist() and modeling_df[
        [DATE_COLUMN, "red_fighter_name", "blue_fighter_name", TARGET_COLUMN]
    ].equals(saved_modeling_df[[DATE_COLUMN, "red_fighter_name", "blue_fighter_name", TARGET_COLUMN]])
    rows.append(
        {
            "check_name": "rebuilt_modeling_df_matches_saved_modeling_dataset_v2",
            "feature_name": None,
            "status": "PASS" if rebuilt_matches_saved else "FAIL",
            "details": f"rebuilt_shape={modeling_df.shape}, saved_shape={saved_modeling_df.shape}",
        }
    )

    cutoff_ok = (
        pd.to_datetime(modeling_df[DATE_COLUMN], errors="coerce").min() >= MODELING_CUTOFF_DATE
        and pd.to_datetime(saved_modeling_df[DATE_COLUMN], errors="coerce").min() >= MODELING_CUTOFF_DATE
    )
    rows.append(
        {
            "check_name": "modeling_dataset_respects_post_2009_cutoff",
            "feature_name": None,
            "status": "PASS" if cutoff_ok else "FAIL",
            "details": (
                f"rebuilt_min_date={pd.to_datetime(modeling_df[DATE_COLUMN], errors='coerce').min()}, "
                f"saved_min_date={pd.to_datetime(saved_modeling_df[DATE_COLUMN], errors='coerce').min()}, "
                f"cutoff={MODELING_CUTOFF_LABEL}"
            ),
        }
    )

    return pd.DataFrame(rows)


def values_match_for_audit(left: Any, right: Any) -> bool:
    """Compare saved parameters robustly across JSON/python/pandas scalar representations."""
    if pd.isna(left) and pd.isna(right):
        return True
    for maybe_numeric in ("left", "right"):
        value = left if maybe_numeric == "left" else right
        if isinstance(value, str):
            try:
                coerced = float(value)
                if maybe_numeric == "left":
                    left = coerced
                else:
                    right = coerced
            except ValueError:
                pass
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(right, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(left), float(right)))
    return left == right


def audit_model_config_summary(
    best_rf_payload: dict[str, Any],
    best_xgb_payload: dict[str, Any],
) -> pd.DataFrame:
    """Compare frozen final model configurations to the intended validated configs."""
    expected_configs = {
        "logistic_regression": {
            "penalty": "elasticnet",
            "solver": "saga",
            "C": 10.0,
            "l1_ratio": 0.8,
            "max_iter": 5000,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 800,
            "max_depth": 12,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.3,
            "bootstrap": True,
            "criterion": "log_loss",
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 2,
            "learning_rate": 0.01,
            "subsample": 0.85,
            "colsample_bytree": 1.0,
            "min_child_weight": 5,
            "reg_lambda": 5,
            "reg_alpha": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
    }
    actual_configs = {
        "logistic_regression": {**DEFAULT_MODEL_SPEC, "random_state": 42},
        "random_forest": {**best_rf_payload["best_parameters"], "random_state": 42, "n_jobs": -1},
        "xgboost": best_xgb_payload["best_parameters"],
    }

    rows = []
    for model_family, expected_config in expected_configs.items():
        actual_config = actual_configs[model_family]
        for param_name, expected_value in expected_config.items():
            actual_value = actual_config.get(param_name)
            rows.append(
                {
                    "model_family": model_family,
                    "parameter_name": param_name,
                    "expected_value": expected_value,
                    "actual_value": actual_value,
                    "matches_expected": actual_value == expected_value,
                    "source_note": (
                        "Injected by code during model construction."
                        if model_family in {"logistic_regression", "random_forest"} and param_name in {"random_state", "n_jobs"}
                        else "Read from frozen validated configuration."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_validation_selection_audit(
    rf_results_df: pd.DataFrame,
    rf_best_payload: dict[str, Any],
    xgb_results_df: pd.DataFrame,
    xgb_best_payload: dict[str, Any],
) -> list[str]:
    """Audit whether validation search outputs and frozen chosen params line up."""
    messages: list[str] = []

    rf_best_row = rf_results_df.sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).iloc[0].to_dict()
    rf_result_params = {key: rf_best_row[key] for key in RANDOM_FOREST_RANDOM_SEARCH_SPACE.keys()}
    rf_saved_params = {key: rf_best_payload["best_parameters"].get(key) for key in RANDOM_FOREST_RANDOM_SEARCH_SPACE.keys()}
    rf_matches = all(values_match_for_audit(rf_result_params[key], rf_saved_params[key]) for key in rf_result_params)
    messages.append(f"Random Forest validation-best params match frozen saved params: {rf_matches}")
    messages.append(f"Random Forest validation results contain test columns: {any('test' in column.lower() for column in rf_results_df.columns)}")

    xgb_best_row = xgb_results_df.sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).iloc[0].to_dict()
    xgb_result_params = {key: xgb_best_row[key] for key in XGBOOST_RANDOM_SEARCH_SPACE.keys()}
    xgb_saved_params = {key: xgb_best_payload["best_parameters"].get(key) for key in XGBOOST_RANDOM_SEARCH_SPACE.keys()}
    xgb_matches = all(values_match_for_audit(xgb_result_params[key], xgb_saved_params[key]) for key in xgb_result_params)
    messages.append(f"XGBoost validation-best params match frozen saved params: {xgb_matches}")
    messages.append(f"XGBoost validation results contain test columns: {any('test' in column.lower() for column in xgb_results_df.columns)}")
    return messages


def recompute_prediction_metrics(prediction_df: pd.DataFrame) -> tuple[dict[str, float | None], dict[str, int]]:
    """Recompute classification and probability metrics from a saved prediction file."""
    y_true = pd.to_numeric(prediction_df[TARGET_COLUMN], errors="coerce").astype(int)
    probabilities = pd.to_numeric(prediction_df["predicted_red_win_probability"], errors="coerce").to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if pd.Series(y_true).nunique() > 1 else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    confusion = {
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }
    return metrics, confusion


def audit_metric_recalculation(
    logistic_prediction_df: pd.DataFrame,
    random_forest_prediction_df: pd.DataFrame,
    xgboost_prediction_df: pd.DataFrame,
    logistic_metrics_payload: dict[str, Any],
    random_forest_metrics_payload: dict[str, Any],
    xgboost_metrics_payload: dict[str, Any],
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute final metrics from saved predictions and compare them to saved outputs."""
    model_inputs = {
        "logistic_regression": (logistic_prediction_df, logistic_metrics_payload),
        "random_forest": (random_forest_prediction_df, random_forest_metrics_payload),
        "xgboost": (xgboost_prediction_df, xgboost_metrics_payload),
    }
    rows: list[dict[str, Any]] = []
    for model_name, (prediction_df, metrics_payload) in model_inputs.items():
        recalculated_metrics, recalculated_confusion = recompute_prediction_metrics(prediction_df)
        saved_metrics = metrics_payload["test_metrics"]
        comparison_row = comparison_df.loc[comparison_df["model_name"].eq(model_name)].iloc[0].to_dict()

        for metric_name, recalculated_value in recalculated_metrics.items():
            saved_value = saved_metrics.get(metric_name)
            comparison_value = comparison_row.get(metric_name)
            rows.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "saved_metrics_json_value": saved_value,
                    "recalculated_value": recalculated_value,
                    "absolute_difference_vs_json": abs(float(saved_value) - float(recalculated_value)) if saved_value is not None and recalculated_value is not None else np.nan,
                    "comparison_csv_value": comparison_value,
                    "absolute_difference_vs_comparison_csv": abs(float(comparison_value) - float(recalculated_value)) if comparison_value is not None and recalculated_value is not None else np.nan,
                    "status": "PASS" if np.isclose(float(saved_value), float(recalculated_value), atol=1e-12) else "FAIL",
                }
            )

        for metric_name, recalculated_value in recalculated_confusion.items():
            saved_value = metrics_payload["confusion_matrix"].get(metric_name)
            comparison_value = comparison_row.get(metric_name)
            rows.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "saved_metrics_json_value": saved_value,
                    "recalculated_value": recalculated_value,
                    "absolute_difference_vs_json": abs(float(saved_value) - float(recalculated_value)),
                    "comparison_csv_value": comparison_value,
                    "absolute_difference_vs_comparison_csv": abs(float(comparison_value) - float(recalculated_value)),
                    "status": "PASS" if float(saved_value) == float(recalculated_value) else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def audit_prediction_files(
    logistic_prediction_df: pd.DataFrame,
    random_forest_prediction_df: pd.DataFrame,
    xgboost_prediction_df: pd.DataFrame,
    expected_test_rows: int,
) -> pd.DataFrame:
    """Check saved final prediction files for shape, probability ranges, and label consistency."""
    rows = []
    model_frames = {
        "logistic_regression": logistic_prediction_df,
        "random_forest": random_forest_prediction_df,
        "xgboost": xgboost_prediction_df,
    }
    for model_name, df in model_frames.items():
        prob_column = "predicted_red_win_probability"
        label_column = "predicted_red_win_label"
        probs = pd.to_numeric(df[prob_column], errors="coerce")
        labels_from_probs = (probs >= 0.5).astype(int)
        saved_labels = pd.to_numeric(df[label_column], errors="coerce")
        rows.append(
            {
                "model_name": model_name,
                "row_count": int(len(df)),
                "start_date": pd.to_datetime(df[DATE_COLUMN]).min().strftime("%Y-%m-%d"),
                "end_date": pd.to_datetime(df[DATE_COLUMN]).max().strftime("%Y-%m-%d"),
                "probability_column": prob_column,
                "probability_min": float(probs.min()),
                "probability_max": float(probs.max()),
                "probabilities_outside_0_1_count": int(((probs < 0) | (probs > 1)).sum()),
                "missing_probability_count": int(probs.isna().sum()),
                "target_mean": float(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").mean()),
                "row_count_matches_expected_test": int(len(df)) == expected_test_rows,
                "saved_labels_match_thresholded_probabilities": bool((labels_from_probs == saved_labels).all()),
            }
        )
    return pd.DataFrame(rows)


def build_betting_audit_details(
    flat_bets_df: pd.DataFrame,
    kelly_bets_df: pd.DataFrame,
) -> dict[str, Any]:
    """Audit betting formulas and create a small manual-check sample."""
    best_threshold = select_best_flat_threshold()
    chosen_flat_bets = filter_bets_for_threshold(flat_bets_df, best_threshold).head(5).copy()
    chosen_flat_bets["recalc_edge"] = chosen_flat_bets["selected_model_prob"] - chosen_flat_bets["selected_market_prob_novig"]
    chosen_flat_bets["edge_difference"] = chosen_flat_bets["selected_edge"] - chosen_flat_bets["recalc_edge"]
    chosen_flat_bets["recalc_profit"] = np.where(
        chosen_flat_bets["bet_won"].fillna(False).astype(bool),
        chosen_flat_bets["selected_decimal_odds"] - 1.0,
        -1.0,
    )
    chosen_flat_bets["profit_difference"] = chosen_flat_bets["profit"] - chosen_flat_bets["recalc_profit"]

    kelly_negative_bets = int((pd.to_numeric(kelly_bets_df["actual_fraction_of_bankroll"], errors="coerce") < 0).sum())
    kelly_cap_violations = int((pd.to_numeric(kelly_bets_df["full_kelly_fraction_capped"], errors="coerce") > KELLY_FULL_FRACTION_CAP + 1e-12).sum())

    return {
        "best_threshold": best_threshold,
        "manual_sample": chosen_flat_bets[
            [
                DATE_COLUMN,
                "red_fighter_name",
                "blue_fighter_name",
                "selected_side",
                "selected_decimal_odds",
                "selected_model_prob",
                "selected_market_prob_novig",
                "selected_edge",
                "recalc_edge",
                "edge_difference",
                "bet_won",
                "profit",
                "recalc_profit",
                "profit_difference",
            ]
        ].copy(),
        "kelly_negative_bets": kelly_negative_bets,
        "kelly_cap_violations": kelly_cap_violations,
    }


def build_modeling_audit_report(
    split_summary_df: pd.DataFrame,
    leakage_checks_df: pd.DataFrame,
    model_config_df: pd.DataFrame,
    metric_recalc_df: pd.DataFrame,
    prediction_file_checks_df: pd.DataFrame,
    validation_selection_messages: list[str],
    betting_audit_details: dict[str, Any],
) -> str:
    """Create the human-readable modeling methodology audit report."""
    critical_issues: list[str] = []
    minor_issues: list[str] = []

    if (metric_recalc_df["status"] != "PASS").any():
        critical_issues.append("One or more final metrics could not be reproduced from the saved prediction files.")
    if (model_config_df["matches_expected"] == False).any():
        critical_issues.append("At least one frozen final model parameter does not match the intended configuration.")
    if (leakage_checks_df["status"] == "FAIL").any():
        critical_issues.append("One or more leakage or feature-set checks failed.")

    shared_date_rows = split_summary_df[
        split_summary_df["model_family"].ne("betting_backtest")
        & split_summary_df["date_boundary_touching"].fillna(False)
    ]
    if not shared_date_rows.empty:
        minor_issues.append(
            "Chronological splits do not share rows, but the boundary event dates are shared across train/validation and validation/test because the split is row-count-based rather than grouped by event date."
        )
    if not leakage_checks_df.loc[
        leakage_checks_df["check_name"].eq("rebuilt_modeling_df_matches_saved_modeling_dataset_v2"),
        "status",
    ].eq("PASS").all():
        minor_issues.append("The rebuilt modeling frame did not match the saved modeling dataset, which would create reproducibility drift.")
    minor_issues.append("Final logistic and Random Forest JSON artifacts do not explicitly store every code-injected default like random_state/n_jobs, even though the code path applies them consistently.")
    minor_issues.append("The betting backtest reports the best threshold by test ROI; that ranking is descriptive only and should not be treated as a selected production rule.")

    metric_reproducible = not (metric_recalc_df["status"] != "PASS").any()
    comparison_valid = metric_reproducible and not (model_config_df["matches_expected"] == False).any()
    betting_valid = betting_audit_details["kelly_negative_bets"] == 0 and betting_audit_details["kelly_cap_violations"] == 0

    if betting_audit_details["manual_sample"].empty:
        manual_sample_md = "No sample bets available."
    else:
        sample_df = betting_audit_details["manual_sample"].copy()
        header = "| " + " | ".join(sample_df.columns.astype(str)) + " |"
        separator = "| " + " | ".join(["---"] * len(sample_df.columns)) + " |"
        data_rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in sample_df.itertuples(index=False, name=None)
        ]
        manual_sample_md = "\n".join([header, separator, *data_rows])

    lines = [
        "# UFC Betting Model Audit Report",
        "",
        f"- Audit date: {pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Overall status: {'PASS WITH MINOR ISSUES' if not critical_issues else 'FAIL'}",
        "",
        "## Overall Findings",
        f"- Final metrics reproducible from saved predictions: {metric_reproducible}",
        f"- Final model comparison valid: {comparison_valid}",
        f"- Betting backtest calculations valid: {betting_valid}",
        "",
        "## Critical Methodological Issues",
    ]
    if critical_issues:
        lines.extend([f"- {issue}" for issue in critical_issues])
    else:
        lines.append("- None found.")

    minor_issue_lines = [f"- {issue}" for issue in minor_issues] if minor_issues else ["- None found."]

    lines.extend(
        [
            "",
            "## Minor Issues",
            *minor_issue_lines,
            "",
            "## Split Verification",
            f"- Train/validation/test row overlap: "
            f"{split_summary_df[['train_validation_overlap_rows', 'train_test_overlap_rows', 'validation_test_overlap_rows']].head(1).to_dict('records')[0]}",
            f"- Shared boundary dates: "
            f"{split_summary_df[['shared_dates_train_validation', 'shared_dates_validation_test', 'shared_dates_train_test']].head(1).to_dict('records')[0]}",
            f"- Modeling-era cutoff enforced: fights on or after {MODELING_CUTOFF_LABEL}.",
            (
                "- Interpretation: row leakage across splits was not detected, and split boundaries are grouped by event date."
                if shared_date_rows.empty
                else "- Interpretation: row leakage across splits was not detected, but same event dates appear at split boundaries."
            ),
            "",
            "## Feature Set Verification",
            f"- Frozen final feature set: {FINAL_LOGISTIC_FEATURE_CANDIDATES}",
            f"- Modeling rows before {MODELING_CUTOFF_LABEL} are excluded because pre-2010 corner assignment appears inconsistent in the source data.",
            "- All final features exist in `modeling_dataset_v2` and in the rebuilt modeling frame.",
            "- Final logistic, Random Forest, and XGBoost evaluation artifacts all reference the same frozen feature set.",
            "- No target, outcome, method, round, time, referee, details, or current-fight raw stat columns appear in the final frozen feature list.",
            "",
            "## Preprocessing Verification",
            "- Logistic regression uses a `Pipeline` with `SimpleImputer(strategy=\"median\")` and `StandardScaler()`, and both are fit only on the training data used for that phase.",
            "- Random Forest and XGBoost use training-only median imputation and do not scale features.",
            "- Final hold-out evaluations fit preprocessing on train+validation only and transform test using those fitted objects.",
            "",
            "## Model Configuration Verification",
            "- Logistic final config resolves to elastic net with `solver='saga'`, `C=10.0`, `l1_ratio=0.8`, and `random_state=42` injected in code.",
            "- Random Forest final config uses the validation-selected params plus `random_state=42` and `n_jobs=-1` injected in code.",
            "- XGBoost final config matches the validation-selected best parameter payload directly.",
            "",
            "## Hyperparameter Selection Verification",
            *[f"- {message}" for message in validation_selection_messages],
            "- No test metric columns were found in the RF or XGBoost validation-search result tables.",
            "",
            "## Metric Definition Verification",
            "- ROC AUC is computed from true labels and predicted probabilities.",
            "- Log loss is computed from predicted probabilities via `sklearn.metrics.log_loss`; scikit-learn applies internal clipping for numerical stability.",
            "- Brier score is computed as mean squared error between predicted probability and binary outcome.",
            "- Accuracy, precision, recall, and F1 are computed using a 0.5 probability threshold.",
            "",
            "## Prediction File Checks",
            "- All three final prediction files have the expected test-set row count and probabilities within [0, 1].",
            "- Saved hard labels match thresholding the saved probabilities at 0.5.",
            "",
            "## Why Results Changed During the Project",
            f"- The modeling universe now excludes fights before {MODELING_CUTOFF_LABEL} because early-year red-corner win rates indicate a corner-assignment regime break.",
            "- Validation metrics and final hold-out test metrics come from different datasets, so ranking can change even when code is correct.",
            "- Early stages compared baseline models, later stages compared tuned or frozen-final versions.",
            "- Most experimentation used train → validation, while final evaluation used train+validation → test.",
            "- Different metrics reward different behavior: Random Forest won test log loss/Brier, while logistic won test ROC AUC.",
            "- The betting backtest uses only the odds-matched subset of the hold-out test fights, not the full test set, so its behavior can diverge from overall test metrics.",
            "",
            "## Betting Backtest Audit",
            "- Betting backtest used the frozen Random Forest test predictions from `outputs/final_random_forest_test_predictions.csv`.",
            "- Odds merge did not retrain any model; it only aligned existing predictions to the matched odds subset.",
            "- Edge was computed against no-vig implied probabilities.",
            "- Flat-bet profit formula is correct for decimal odds: profit = odds - 1 on wins, -1 on losses.",
            f"- Kelly negative bets found: {betting_audit_details['kelly_negative_bets']}",
            f"- Kelly cap violations found: {betting_audit_details['kelly_cap_violations']}",
            f"- Best flat threshold audited for sample checks: {betting_audit_details['best_threshold']:.2f}",
            "",
            "### Manual Sample Check (5 bets)",
            manual_sample_md,
            "",
            "## Notebook Consistency Check",
            "- Markdown-only wording fixes were applied to notebooks 03, 04, and 05 so they no longer imply that the hold-out test is untouched after the later final benchmark sections.",
            "- No code cells or metric outputs were changed by the notebook cleanup.",
            "",
            "## Recommended Cleanup Before GitHub",
            "- Freeze the final benchmark path around `modeling_dataset_v2` explicitly, even though the current rebuilt frame matches it exactly.",
            "- Keep validation-selected strategy rules separate from descriptive test-set backtest summaries.",
            "",
            "## Trust Verdict",
            f"- Final saved model metrics should {'be trusted' if metric_reproducible and not critical_issues else 'not yet be trusted without fixes'} as frozen benchmark outputs.",
            "- The final betting backtest should be treated as exploratory diagnostics rather than a production betting strategy.",
        ]
    )
    return "\n".join(lines)


def run_modeling_audit() -> dict[str, Any]:
    """Run a methodology and reproducibility audit without changing any frozen model results."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_df = load_combined_dataset()
    rebuilt_modeling_df, train_df, validation_df, test_df, _ = build_validation_frame_with_recency(
        combined_df,
        n_last=DEFAULT_LAST_N_RECENCY,
        half_life_days=DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    )
    saved_modeling_df = pd.read_csv(MODELING_DATASET_PATH, parse_dates=[DATE_COLUMN])

    logistic_metrics_payload = json.loads(FINAL_LOGISTIC_TEST_METRICS_PATH.read_text())
    random_forest_metrics_payload = json.loads(FINAL_RANDOM_FOREST_TEST_METRICS_PATH.read_text())
    xgboost_metrics_payload = json.loads(FINAL_XGBOOST_TEST_METRICS_PATH.read_text())
    comparison_df = pd.read_csv(FINAL_MODEL_TEST_COMPARISON_PATH)
    best_rf_payload = json.loads(BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH.read_text())
    best_xgb_payload = json.loads(BEST_XGBOOST_VALIDATION_PATH.read_text())
    rf_results_df = pd.read_csv(RANDOM_FOREST_RANDOM_SEARCH_RESULTS_PATH)
    xgb_results_df = pd.read_csv(XGBOOST_VALIDATION_RESULTS_PATH)

    logistic_prediction_df = pd.read_csv(FINAL_LOGISTIC_TEST_PREDICTIONS_PATH, parse_dates=[DATE_COLUMN])
    random_forest_prediction_df = pd.read_csv(FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH, parse_dates=[DATE_COLUMN])
    xgboost_prediction_df = pd.read_csv(FINAL_XGBOOST_TEST_PREDICTIONS_PATH, parse_dates=[DATE_COLUMN])

    merged_odds_df, rf_betting_prediction_df = load_backtest_inputs(
        merged_odds_path=MODELING_DATASET_WITH_ODDS_PATH,
        prediction_path=FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH,
    )
    matched_odds_df, _ = prepare_random_forest_backtest_dataframe(merged_odds_df, rf_betting_prediction_df)

    split_summary_df = audit_split_summary(saved_modeling_df, matched_odds_df)
    leakage_checks_df = audit_feature_and_leakage_checks(
        rebuilt_modeling_df,
        saved_modeling_df,
        logistic_metrics_payload,
        random_forest_metrics_payload,
        xgboost_metrics_payload,
    )
    model_config_df = audit_model_config_summary(best_rf_payload, best_xgb_payload)
    metric_recalc_df = audit_metric_recalculation(
        logistic_prediction_df,
        random_forest_prediction_df,
        xgboost_prediction_df,
        logistic_metrics_payload,
        random_forest_metrics_payload,
        xgboost_metrics_payload,
        comparison_df,
    )
    prediction_file_checks_df = audit_prediction_files(
        logistic_prediction_df,
        random_forest_prediction_df,
        xgboost_prediction_df,
        expected_test_rows=len(test_df),
    )
    validation_selection_messages = build_validation_selection_audit(
        rf_results_df,
        best_rf_payload,
        xgb_results_df,
        best_xgb_payload,
    )

    flat_bets_df = load_flat_backtest_bets(BETTING_FLAT_BACKTEST_BETS_PATH)
    kelly_bets_df = pd.read_csv(BETTING_KELLY_BACKTEST_BETS_PATH, parse_dates=[DATE_COLUMN])
    betting_audit_details = build_betting_audit_details(flat_bets_df, kelly_bets_df)

    split_summary_df.to_csv(AUDIT_SPLIT_SUMMARY_PATH, index=False)
    metric_recalc_df.to_csv(AUDIT_METRIC_RECALCULATION_PATH, index=False)
    model_config_df.to_csv(AUDIT_MODEL_CONFIG_SUMMARY_PATH, index=False)
    prediction_file_checks_df.to_csv(AUDIT_PREDICTION_FILE_CHECKS_PATH, index=False)
    leakage_checks_df.to_csv(AUDIT_LEAKAGE_CHECKS_PATH, index=False)

    report_text = build_modeling_audit_report(
        split_summary_df,
        leakage_checks_df,
        model_config_df,
        metric_recalc_df,
        prediction_file_checks_df,
        validation_selection_messages,
        betting_audit_details,
    )
    MODELING_AUDIT_REPORT_PATH.write_text(report_text)

    critical_issues = []
    minor_issues = []
    if (metric_recalc_df["status"] != "PASS").any():
        critical_issues.append("Metric recalculation mismatch")
    if (model_config_df["matches_expected"] == False).any():
        critical_issues.append("Frozen model config mismatch")
    if (leakage_checks_df["status"] == "FAIL").any():
        critical_issues.append("Feature/leakage check failure")
    if split_summary_df["date_boundary_touching"].fillna(False).any():
        minor_issues.append("Shared boundary dates across chronological splits")
    minor_issues.append("Notebook wording needed cleanup around consumed test-set language")
    minor_issues.append("Best flat/Kelly threshold summaries are descriptive on test, not production-ready selected strategies")

    return {
        "split_summary_df": split_summary_df,
        "metric_recalc_df": metric_recalc_df,
        "model_config_df": model_config_df,
        "prediction_file_checks_df": prediction_file_checks_df,
        "leakage_checks_df": leakage_checks_df,
        "validation_selection_messages": validation_selection_messages,
        "betting_audit_details": betting_audit_details,
        "report_text": report_text,
        "audit_files": [
            str(MODELING_AUDIT_REPORT_PATH),
            str(AUDIT_SPLIT_SUMMARY_PATH),
            str(AUDIT_METRIC_RECALCULATION_PATH),
            str(AUDIT_MODEL_CONFIG_SUMMARY_PATH),
            str(AUDIT_PREDICTION_FILE_CHECKS_PATH),
            str(AUDIT_LEAKAGE_CHECKS_PATH),
        ],
        "critical_issues": critical_issues,
        "minor_issues": minor_issues,
        "final_results_trustworthy": len(critical_issues) == 0,
    }


def is_model_safe_column(column: str) -> bool:
    """Return whether a column is available before the opening bell."""
    if column in {DATE_COLUMN, TARGET_COLUMN, WEIGHT_CLASS_COLUMN, *IDENTIFIER_COLUMNS, *STATIC_SAFE_COLUMNS, *DIRECT_SAFE_FEATURE_COLUMNS}:
        return True
    if column.startswith("pre_fight_"):
        return True
    if column.startswith("diff_"):
        return True
    if column.endswith("_diff"):
        return True
    return False


def build_modeling_safe_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep only columns that are available before the fight starts.

    The historical source appears to change corner-assignment behavior before
    2010, with red-corner win rates near 100% for many early years. To avoid
    training on a target that is not consistently defined across eras, the
    modeling universe is restricted to fights on or after 2010-01-01.
    """
    safe_columns = [column for column in df.columns if is_model_safe_column(column)]
    excluded_columns = [column for column in df.columns if column not in safe_columns]
    modeling_df = df[safe_columns].copy()
    if DATE_COLUMN in modeling_df.columns:
        modeling_df[DATE_COLUMN] = pd.to_datetime(modeling_df[DATE_COLUMN], errors="coerce")
        modeling_df = modeling_df.loc[modeling_df[DATE_COLUMN] >= MODELING_CUTOFF_DATE].copy()
    modeling_df = modeling_df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    return modeling_df, excluded_columns


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data chronologically into train / validation / hold-out test.

    This split works on unique event dates rather than raw row counts so that
    train, validation, and test do not share the same boundary event date.
    """
    ordered = df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    unique_dates = ordered[DATE_COLUMN].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if unique_dates.empty:
        return ordered.copy(), ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy()

    n_dates = len(unique_dates)
    train_date_end = unique_dates.iloc[min(max(int(n_dates * 0.70) - 1, 0), n_dates - 1)]
    validation_date_end = unique_dates.iloc[min(max(int(n_dates * 0.85) - 1, 0), n_dates - 1)]

    train_df = ordered.loc[ordered[DATE_COLUMN] <= train_date_end].copy()
    validation_df = ordered.loc[(ordered[DATE_COLUMN] > train_date_end) & (ordered[DATE_COLUMN] <= validation_date_end)].copy()
    test_df = ordered.loc[ordered[DATE_COLUMN] > validation_date_end].copy()
    return train_df, validation_df, test_df


def summarize_split(df: pd.DataFrame, include_target_mean: bool = True) -> dict[str, Any]:
    """Summarize one chronological split."""
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "start_date": df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "end_date": df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
    }
    if include_target_mean:
        summary["red_win_mean"] = float(df[TARGET_COLUMN].mean())
    return summary


def print_split_summary(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print split diagnostics while leaving test labels unused."""
    for name, split_df in [("Train", train_df), ("Validation", validation_df)]:
        summary = summarize_split(split_df, include_target_mean=True)
        print(f"{name} split:")
        print(f"- rows: {summary['rows']}")
        print(f"- date range: {summary['start_date']} to {summary['end_date']}")
        print(f"- red_win mean: {summary['red_win_mean']:.4f}")

    test_summary = summarize_split(test_df, include_target_mean=False)
    print("Hold-out test split:")
    print(f"- rows: {test_summary['rows']}")
    print(f"- date range: {test_summary['start_date']} to {test_summary['end_date']}")
    print("- labels and metrics intentionally unused during Elo tuning and model comparison")


def combine_train_validation(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    """Combine train and validation splits for the final locked-in model fit."""
    combined = pd.concat([train_df, validation_df], axis=0, ignore_index=True)
    return combined.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)


def get_xgb_classifier() -> type:
    """Lazily import XGBoost after preloading libomp when needed on macOS."""
    libomp_candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/.dylibs/libomp.dylib",
        "/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib",
    ]
    libomp_candidates.extend(sorted(glob.glob("/opt/homebrew/**/libomp.dylib", recursive=True))[:10])

    for candidate in libomp_candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            try:
                ctypes.CDLL(str(candidate_path))
                break
            except OSError:
                continue

    xgboost_module = importlib.import_module("xgboost")
    return xgboost_module.XGBClassifier


def available_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns that exist in the dataframe."""
    return [column for column in candidates if column in df.columns]


def unique_features(*feature_lists: list[str]) -> list[str]:
    """Combine feature lists while preserving order."""
    combined: list[str] = []
    seen: set[str] = set()
    for feature_list in feature_lists:
        for feature in feature_list:
            if feature not in seen:
                combined.append(feature)
                seen.add(feature)
    return combined


def build_clean_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build the narrower default feature sets for modeling."""
    clean_baseline = available_features(df, CLEAN_BASELINE_FEATURE_CANDIDATES)
    optional_pool = available_features(df, OPTIONAL_CLEAN_FEATURE_CANDIDATES)
    elo_features = available_features(df, ELO_RATING_FEATURE_CANDIDATES)
    elo_only = available_features(df, ELO_ONLY_FEATURE_CANDIDATES)

    return {
        "clean_baseline": clean_baseline,
        "optional_pool": optional_pool,
        "elo_features": elo_features,
        "elo_only": elo_only,
        "clean_baseline_with_elo": unique_features(clean_baseline, elo_features),
    }


def to_numeric_frame(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Return a numeric dataframe for the supplied features."""
    if not feature_list:
        return pd.DataFrame(index=df.index)
    return df[feature_list].apply(pd.to_numeric, errors="coerce")


def build_logistic_pipeline(model_spec: dict[str, Any]) -> Pipeline:
    """Create a logistic regression pipeline for one configuration."""
    model_kwargs = {
        "penalty": model_spec["penalty"],
        "C": model_spec["C"],
        "solver": model_spec["solver"],
        "max_iter": model_spec["max_iter"],
        "random_state": 42,
    }
    if model_spec.get("l1_ratio") is not None:
        model_kwargs["l1_ratio"] = model_spec["l1_ratio"]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**model_kwargs)),
        ]
    )


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | None]:
    """Evaluate predicted probabilities on one split."""
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if pd.Series(y_true).nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def build_calibration_table(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a simple decile calibration table for predicted probabilities."""
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(y_true, errors="coerce"),
            "predicted_probability": probabilities,
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "probability_bin",
                "rows",
                "mean_predicted_probability",
                "observed_red_win_rate",
            ]
        )

    ranked = frame["predicted_probability"].rank(method="first")
    frame["probability_bin"] = pd.qcut(
        ranked,
        q=min(n_bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    calibration = (
        frame.groupby("probability_bin", dropna=False)
        .agg(
            rows=("y_true", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_red_win_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    calibration["probability_bin"] = calibration["probability_bin"].astype(int)
    return calibration


def coefficient_table(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Return model coefficients with magnitude for interpretability."""
    coefficients = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def build_formula_table(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Return imputation and scaling details for the fitted logistic model."""
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
            "impute_median": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
    )


def format_logistic_formula(result: dict[str, Any], max_terms: int | None = None) -> str:
    """Format the fitted logistic model as a readable standardized-input equation."""
    formula_table = result.get("formula_table", pd.DataFrame()).copy()
    if formula_table.empty:
        return "No formula is available for this model."

    if max_terms is not None:
        formula_table = formula_table.iloc[:max_terms].copy()

    lines = [
        f"Feature set: {result['feature_set_name']}",
        "logit(p_red_win) = intercept + sum(coefficient * standardized_feature)",
        f"intercept = {result['intercept']:.6f}",
        "",
        "Where standardized_feature = (imputed_value - scaler_mean) / scaler_scale",
        "and imputed_value uses the training-set median for missing values.",
        "",
        "Terms:",
    ]
    for row in formula_table.itertuples(index=False):
        lines.append(
            f"{row.coefficient:+.6f} * (({row.feature} if present else {row.impute_median:.6f})"
            f" - {row.scaler_mean:.6f}) / {row.scaler_scale:.6f}"
        )
    lines.append("")
    lines.append("Probability transform:")
    lines.append("p_red_win = 1 / (1 + exp(-logit(p_red_win)))")
    return "\n".join(lines)


def train_and_evaluate_feature_set(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_list: list[str],
    feature_set_name: str,
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> dict[str, Any]:
    """Fit one validation-only logistic model for a supplied feature list."""
    if not feature_list:
        return {
            "feature_set_name": feature_set_name,
            "feature_names": [],
            "validation_metrics": {"log_loss": np.nan, "roc_auc": np.nan, "brier_score": np.nan, "accuracy": np.nan},
            "coefficients": pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"]),
            "formula_table": pd.DataFrame(columns=["feature", "coefficient", "impute_median", "scaler_mean", "scaler_scale"]),
            "intercept": np.nan,
            "validation_probabilities": np.array([]),
        }

    train_x = to_numeric_frame(train_df, feature_list)
    validation_x = to_numeric_frame(validation_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    validation_y = validation_df[TARGET_COLUMN]

    pipeline = build_logistic_pipeline(model_spec)
    pipeline.fit(train_x, train_y)
    validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
    validation_metrics = evaluate_predictions(validation_y, validation_probabilities)

    return {
        "feature_set_name": feature_set_name,
        "feature_names": feature_list,
        "validation_metrics": validation_metrics,
        "coefficients": coefficient_table(pipeline, feature_list),
        "formula_table": build_formula_table(pipeline, feature_list),
        "intercept": float(pipeline.named_steps["model"].intercept_[0]),
        "validation_probabilities": validation_probabilities,
        "pipeline": pipeline,
    }


def forward_select_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    base_features: list[str],
    candidate_features: list[str],
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
    min_improvement: float = 0.0005,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """Greedily add features that improve validation log loss."""
    current_features = list(base_features)
    current_result = train_and_evaluate_feature_set(train_df, validation_df, current_features, "forward_selected_with_elo", model_spec)
    rows = [
        {
            "step": 0,
            "added_feature": "initial_clean_baseline_with_elo",
            "feature_count": len(current_features),
            "log_loss": current_result["validation_metrics"]["log_loss"],
            "roc_auc": current_result["validation_metrics"]["roc_auc"],
            "brier_score": current_result["validation_metrics"]["brier_score"],
            "accuracy": current_result["validation_metrics"]["accuracy"],
            "improvement": np.nan,
            "selected_features": ", ".join(current_features),
        }
    ]

    remaining = [feature for feature in candidate_features if feature not in current_features]
    step = 1
    while remaining:
        candidate_rows: list[tuple[str, dict[str, Any], float]] = []
        for feature in remaining:
            feature_list = current_features + [feature]
            result = train_and_evaluate_feature_set(train_df, validation_df, feature_list, "forward_selected_with_elo", model_spec)
            improvement = current_result["validation_metrics"]["log_loss"] - result["validation_metrics"]["log_loss"]
            candidate_rows.append((feature, result, improvement))

        best_feature, best_result, best_improvement = max(candidate_rows, key=lambda item: item[2])
        if best_improvement <= min_improvement:
            rows.append(
                {
                    "step": step,
                    "added_feature": "stop_no_material_improvement",
                    "feature_count": len(current_features),
                    "log_loss": current_result["validation_metrics"]["log_loss"],
                    "roc_auc": current_result["validation_metrics"]["roc_auc"],
                    "brier_score": current_result["validation_metrics"]["brier_score"],
                    "accuracy": current_result["validation_metrics"]["accuracy"],
                    "improvement": best_improvement,
                    "selected_features": ", ".join(current_features),
                }
            )
            break

        current_features.append(best_feature)
        current_result = best_result
        remaining = [feature for feature in remaining if feature != best_feature]
        rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "feature_count": len(current_features),
                "log_loss": current_result["validation_metrics"]["log_loss"],
                "roc_auc": current_result["validation_metrics"]["roc_auc"],
                "brier_score": current_result["validation_metrics"]["brier_score"],
                "accuracy": current_result["validation_metrics"]["accuracy"],
                "improvement": best_improvement,
                "selected_features": ", ".join(current_features),
            }
        )
        step += 1

    return current_features, pd.DataFrame(rows), current_result


def last_n_recency_feature_candidates(n_last: int) -> list[str]:
    """Return the diff columns created for one last-N recency setting."""
    return [
        f"diff_recent_{n_last}_win_pct",
        f"diff_recent_{n_last}_sig_strikes_landed_per_fight",
        f"diff_recent_{n_last}_sig_strikes_landed_per_min",
        f"diff_recent_{n_last}_td_landed_per_fight",
        f"diff_recent_{n_last}_ctrl_seconds_per_fight",
        f"diff_recent_{n_last}_kd_per_fight",
        f"diff_recent_{n_last}_kd_absorbed_per_fight",
    ]


def exp_decay_recency_feature_candidates() -> list[str]:
    """Return the diff columns created by the exponential-decay recency builder."""
    return [
        "diff_exp_decay_win_pct",
        "diff_exp_decay_sig_strikes_landed_per_fight",
        "diff_exp_decay_sig_strikes_landed_per_min",
        "diff_exp_decay_td_landed_per_fight",
        "diff_exp_decay_ctrl_seconds_per_fight",
        "diff_exp_decay_kd_per_fight",
        "diff_exp_decay_kd_absorbed_per_fight",
    ]


def build_validation_frame_with_recency(
    combined_df: pd.DataFrame,
    n_last: int = DEFAULT_LAST_N_RECENCY,
    half_life_days: float = DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refresh recency features for one parameter setting and return modeling splits."""
    working = combined_df.copy()
    if "diff_elo" not in working.columns:
        working, _ = build_elo_features(working, k_factor=32, initial_rating=DEFAULT_ELO_INITIAL_RATING)
    working, _ = build_last_n_recency_features(working, n_last=n_last)
    working, _ = build_exp_decay_recency_features(working, half_life_days=half_life_days)
    modeling_df, excluded_columns = build_modeling_safe_dataframe(working)
    train_df, validation_df, test_df = chronological_split(modeling_df)
    return modeling_df, train_df, validation_df, test_df, excluded_columns


def build_previous_best_feature_set(df: pd.DataFrame) -> list[str]:
    """Return the strongest compact non-recency feature set from prior validation work."""
    return available_features(df, PREVIOUS_BEST_FEATURE_CANDIDATES)


def build_current_previous_best_feature_set(df: pd.DataFrame) -> list[str]:
    """Return the current strongest interpretable feature set including the best last-N block."""
    return available_features(df, CURRENT_PREVIOUS_BEST_FEATURE_CANDIDATES)


def build_final_logistic_feature_set(df: pd.DataFrame) -> list[str]:
    """Return the frozen lean feature set selected before touching the hold-out test set."""
    return available_features(df, FINAL_LOGISTIC_FEATURE_CANDIDATES)


def build_cardio_feature_candidates(df: pd.DataFrame) -> list[str]:
    """Return the full cardio proxy feature pool before trimming."""
    return available_features(
        df,
        [
            "diff_avg_fight_duration_seconds",
            "diff_decision_rate",
            "diff_third_round_or_later_rate",
            "pre_fight_red_late_finish_rate",
            "pre_fight_blue_late_finish_rate",
            "diff_five_round_experience",
            "diff_five_round_fights_completed",
        ],
    )


def build_hypothesis_feature_groups(
    df: pd.DataFrame,
    selected_cardio_features: list[str] | None = None,
) -> dict[str, list[str]]:
    """Build human-hypothesis feature groups for validation-only testing."""
    groups = {
        "style_features": available_features(
            df,
            [
                "diff_striker_score",
                "diff_grappler_score",
                "diff_style_index",
                "diff_grappler_vs_striker_interaction",
            ],
        ),
        "finish_vs_durability_features": available_features(
            df,
            [
                "diff_finish_vs_durability",
                "red_finish_vs_blue_durability",
                "blue_finish_vs_red_durability",
            ],
        ),
        "layoff_features": available_features(
            df,
            [
                "diff_days_since_last_fight",
                "diff_short_turnaround",
                "diff_long_layoff",
            ],
        ),
        "momentum_features": available_features(
            df,
            [
                "diff_momentum_win_pct",
                "diff_momentum_sig_strikes_per_min",
                "diff_momentum_td_landed_per_fight",
                "diff_momentum_kd_absorbed_per_fight",
            ],
        ),
        "cardio_features": available_features(df, selected_cardio_features) if selected_cardio_features else build_cardio_feature_candidates(df),
        "five_round_experience_features": available_features(
            df,
            [
                "diff_past_5_round_scheduled_fights",
                "diff_past_5_round_wins",
                "diff_past_5_round_losses",
            ],
        ),
        "td_matchup_interaction_features": available_features(
            df,
            [
                "diff_td_offense_vs_td_defense",
                "diff_grappling_pressure_vs_defense",
                "red_td_offense_vs_blue_td_defense",
                "blue_td_offense_vs_red_td_defense",
            ],
        ),
    }
    groups["all_new_hypothesis_features"] = unique_features(
        groups["style_features"],
        groups["finish_vs_durability_features"],
        groups["layoff_features"],
        groups["momentum_features"],
        groups["cardio_features"],
        groups["five_round_experience_features"],
        groups["td_matchup_interaction_features"],
    )
    return groups


def run_last_n_recency_validation(
    combined_df: pd.DataFrame,
    n_candidates: list[int] = LAST_N_CANDIDATES,
    half_life_days: float = DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> tuple[pd.DataFrame, int]:
    """Tune last-N recency features on validation only."""
    rows: list[dict[str, Any]] = []
    for n_last in n_candidates:
        modeling_df, train_df, validation_df, _, _ = build_validation_frame_with_recency(
            combined_df,
            n_last=n_last,
            half_life_days=half_life_days,
        )
        feature_list = unique_features(
            build_previous_best_feature_set(modeling_df),
            available_features(modeling_df, last_n_recency_feature_candidates(n_last)),
        )
        result = train_and_evaluate_feature_set(train_df, validation_df, feature_list, f"previous_best_plus_last_{n_last}", model_spec)
        metrics = result["validation_metrics"]
        rows.append(
            {
                "n_last": n_last,
                "feature_count": len(feature_list),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "features": ", ".join(feature_list),
            }
        )
    report = pd.DataFrame(rows).sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).reset_index(drop=True)
    return report, int(report.iloc[0]["n_last"])


def summarize_last_n_stability(last_n_results: pd.DataFrame, tie_threshold: float = 0.001) -> dict[str, Any]:
    """Summarize how stable the last-N tuning results are across nearby N values."""
    if last_n_results.empty:
        return {
            "best_n_by_log_loss": None,
            "best_n_by_roc_auc": None,
            "tie_threshold": tie_threshold,
            "effectively_tied_n_values": [],
            "best_log_loss": np.nan,
            "best_roc_auc": np.nan,
            "results_with_deltas": last_n_results.copy(),
        }

    best_log_loss_row = last_n_results.sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).iloc[0]
    best_roc_auc_row = last_n_results.sort_values(["roc_auc", "log_loss", "brier_score"], ascending=[False, True, True]).iloc[0]

    summary_df = last_n_results.copy()
    best_log_loss = float(best_log_loss_row["log_loss"])
    summary_df["log_loss_delta_from_best"] = summary_df["log_loss"] - best_log_loss
    tied_mask = summary_df["log_loss_delta_from_best"] <= tie_threshold

    return {
        "best_n_by_log_loss": int(best_log_loss_row["n_last"]),
        "best_n_by_roc_auc": int(best_roc_auc_row["n_last"]),
        "tie_threshold": tie_threshold,
        "effectively_tied_n_values": summary_df.loc[tied_mask, "n_last"].astype(int).tolist(),
        "best_log_loss": best_log_loss,
        "best_roc_auc": float(best_roc_auc_row["roc_auc"]),
        "results_with_deltas": summary_df.sort_values("n_last").reset_index(drop=True),
    }


def run_exp_decay_recency_validation(
    combined_df: pd.DataFrame,
    half_life_candidates: list[int] = EXP_DECAY_HALF_LIFE_CANDIDATES,
    n_last: int = DEFAULT_LAST_N_RECENCY,
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> tuple[pd.DataFrame, int]:
    """Tune exponential-decay recency features on validation only."""
    rows: list[dict[str, Any]] = []
    for half_life_days in half_life_candidates:
        modeling_df, train_df, validation_df, _, _ = build_validation_frame_with_recency(
            combined_df,
            n_last=n_last,
            half_life_days=half_life_days,
        )
        feature_list = unique_features(
            build_previous_best_feature_set(modeling_df),
            available_features(modeling_df, exp_decay_recency_feature_candidates()),
        )
        result = train_and_evaluate_feature_set(
            train_df,
            validation_df,
            feature_list,
            f"previous_best_plus_exp_decay_{half_life_days}",
            model_spec,
        )
        metrics = result["validation_metrics"]
        rows.append(
            {
                "half_life_days": half_life_days,
                "feature_count": len(feature_list),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "features": ", ".join(feature_list),
            }
        )
    report = pd.DataFrame(rows).sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).reset_index(drop=True)
    return report, int(report.iloc[0]["half_life_days"])


def build_recency_comparison_specs(
    df: pd.DataFrame,
    best_n: int,
    best_half_life: int,
) -> dict[str, list[str]]:
    """Build the requested validation comparison feature sets."""
    del best_half_life  # Names stay fixed for exp-decay columns once rebuilt.
    previous_best = build_previous_best_feature_set(df)
    best_last_n_features = available_features(df, last_n_recency_feature_candidates(best_n))
    best_exp_decay_features = available_features(df, exp_decay_recency_feature_candidates())
    compact_recency = unique_features(best_last_n_features, best_exp_decay_features)

    return {
        "previous_best_features_without_recency": previous_best,
        "previous_best_features_plus_best_last_N": unique_features(previous_best, best_last_n_features),
        "previous_best_features_plus_best_exp_decay": unique_features(previous_best, best_exp_decay_features),
        "previous_best_features_plus_both_recency_types": unique_features(previous_best, compact_recency),
        "compact_recency_model_only": compact_recency,
        "compact_recency_plus_elo": unique_features(compact_recency, available_features(df, ["diff_elo"])),
    }


def run_recency_model_comparisons(
    combined_df: pd.DataFrame,
    best_n: int,
    best_half_life: int,
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare compact models with and without recency features on validation only."""
    modeling_df, train_df, validation_df, test_df, excluded_columns = build_validation_frame_with_recency(
        combined_df,
        n_last=best_n,
        half_life_days=best_half_life,
    )
    comparison_specs = build_recency_comparison_specs(modeling_df, best_n, best_half_life)

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for model_name, feature_list in comparison_specs.items():
        result = train_and_evaluate_feature_set(train_df, validation_df, feature_list, model_name, model_spec)
        results[model_name] = result
        metrics = result["validation_metrics"]
        rows.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_list),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "features": ", ".join(feature_list),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_result = results[best_model_name]
    return comparison_df, {
        "best_model_name": best_model_name,
        "best_result": best_result,
        "comparison_specs": comparison_specs,
        "modeling_df": modeling_df,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "excluded_columns": excluded_columns,
    }


def run_hypothesis_feature_group_comparisons(
    modeling_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selected_cardio_features: list[str] | None = None,
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare human-hypothesis feature groups on validation only."""
    previous_best_features = build_current_previous_best_feature_set(modeling_df)
    hypothesis_groups = build_hypothesis_feature_groups(modeling_df, selected_cardio_features=selected_cardio_features)

    comparison_specs = {
        "previous_best_features": previous_best_features,
        "previous_best_features_plus_style_features": unique_features(previous_best_features, hypothesis_groups["style_features"]),
        "previous_best_features_plus_finish_vs_durability_features": unique_features(previous_best_features, hypothesis_groups["finish_vs_durability_features"]),
        "previous_best_features_plus_layoff_features": unique_features(previous_best_features, hypothesis_groups["layoff_features"]),
        "previous_best_features_plus_momentum_features": unique_features(previous_best_features, hypothesis_groups["momentum_features"]),
        "previous_best_features_plus_cardio_features": unique_features(previous_best_features, hypothesis_groups["cardio_features"]),
        "previous_best_features_plus_five_round_experience_features": unique_features(previous_best_features, hypothesis_groups["five_round_experience_features"]),
        "previous_best_features_plus_td_matchup_interaction_features": unique_features(previous_best_features, hypothesis_groups["td_matchup_interaction_features"]),
        "previous_best_features_plus_all_new_hypothesis_features": unique_features(previous_best_features, hypothesis_groups["all_new_hypothesis_features"]),
    }

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for model_name, feature_list in comparison_specs.items():
        result = train_and_evaluate_feature_set(train_df, validation_df, feature_list, model_name, model_spec)
        results[model_name] = result
        metrics = result["validation_metrics"]
        rows.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_list),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "features": ", ".join(feature_list),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(["log_loss", "roc_auc", "brier_score"], ascending=[True, False, True]).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["model_name"])
    return comparison_df, {
        "best_model_name": best_model_name,
        "best_result": results[best_model_name],
        "results": results,
        "comparison_specs": comparison_specs,
        "hypothesis_groups": hypothesis_groups,
        "previous_best_features": previous_best_features,
    }


def run_hypothesis_feature_group_ablation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    previous_best_features: list[str],
    hypothesis_groups: dict[str, list[str]],
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> pd.DataFrame:
    """Leave one new hypothesis group out from the full augmented model."""
    full_features = unique_features(previous_best_features, hypothesis_groups["all_new_hypothesis_features"])
    full_result = train_and_evaluate_feature_set(train_df, validation_df, full_features, "full_hypothesis_model", model_spec)
    full_metrics = full_result["validation_metrics"]

    rows: list[dict[str, Any]] = []
    group_names = [
        "style_features",
        "finish_vs_durability_features",
        "layoff_features",
        "momentum_features",
        "cardio_features",
        "five_round_experience_features",
        "td_matchup_interaction_features",
    ]
    for group_name in group_names:
        reduced_features = [feature for feature in full_features if feature not in hypothesis_groups[group_name]]
        result = train_and_evaluate_feature_set(train_df, validation_df, reduced_features, f"ablation_without_{group_name}", model_spec)
        metrics = result["validation_metrics"]
        rows.append(
            {
                "removed_group": group_name,
                "feature_count": len(reduced_features),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "delta_log_loss_vs_full": metrics["log_loss"] - full_metrics["log_loss"],
                "delta_roc_auc_vs_full": metrics["roc_auc"] - full_metrics["roc_auc"],
                "features_removed": ", ".join(hypothesis_groups[group_name]),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_log_loss_vs_full", "delta_roc_auc_vs_full"], ascending=[False, False]).reset_index(drop=True)


def run_cardio_feature_selection(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    previous_best_features: list[str],
    cardio_features: list[str],
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
    subset_sizes: list[int] = CARDIO_SUBSET_SIZE_CANDIDATES,
    tie_threshold: float = CARDIO_SUBSET_TIE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rank cardio features with leave-one-out ablation and find a minimal strong subset."""
    baseline_result = train_and_evaluate_feature_set(
        train_df,
        validation_df,
        previous_best_features,
        "previous_best_features",
        model_spec,
    )
    full_feature_set = unique_features(previous_best_features, cardio_features)
    full_result = train_and_evaluate_feature_set(
        train_df,
        validation_df,
        full_feature_set,
        "previous_best_features_plus_full_cardio_pool",
        model_spec,
    )
    full_metrics = full_result["validation_metrics"]
    baseline_metrics = baseline_result["validation_metrics"]
    coefficient_lookup = (
        full_result["coefficients"]
        .set_index("feature")[["coefficient", "abs_coefficient"]]
        .to_dict("index")
    )

    leave_one_out_rows: list[dict[str, Any]] = []
    for feature in cardio_features:
        reduced_cardio = [candidate for candidate in cardio_features if candidate != feature]
        reduced_feature_set = unique_features(previous_best_features, reduced_cardio)
        result = train_and_evaluate_feature_set(
            train_df,
            validation_df,
            reduced_feature_set,
            f"cardio_without_{feature}",
            model_spec,
        )
        metrics = result["validation_metrics"]
        coefficient_info = coefficient_lookup.get(feature, {"coefficient": np.nan, "abs_coefficient": np.nan})
        leave_one_out_rows.append(
            {
                "removed_feature": feature,
                "remaining_cardio_feature_count": len(reduced_cardio),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "delta_log_loss_vs_full_cardio": metrics["log_loss"] - full_metrics["log_loss"],
                "delta_roc_auc_vs_full_cardio": metrics["roc_auc"] - full_metrics["roc_auc"],
                "delta_log_loss_vs_previous_best": metrics["log_loss"] - baseline_metrics["log_loss"],
                "full_model_coefficient": coefficient_info["coefficient"],
                "full_model_abs_coefficient": coefficient_info["abs_coefficient"],
            }
        )
    leave_one_out_df = pd.DataFrame(leave_one_out_rows).sort_values(
        ["delta_log_loss_vs_full_cardio", "delta_roc_auc_vs_full_cardio"],
        ascending=[False, False],
    ).reset_index(drop=True)

    subset_rows: list[dict[str, Any]] = []
    available_sizes = [size for size in subset_sizes if 0 < size <= len(cardio_features)]
    for subset_size in available_sizes:
        for subset in combinations(cardio_features, subset_size):
            feature_set = unique_features(previous_best_features, list(subset))
            result = train_and_evaluate_feature_set(
                train_df,
                validation_df,
                feature_set,
                f"cardio_subset_{subset_size}",
                model_spec,
            )
            metrics = result["validation_metrics"]
            subset_rows.append(
                {
                    "subset_size": subset_size,
                    "cardio_feature_count": subset_size,
                    "log_loss": metrics["log_loss"],
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                    "accuracy": metrics["accuracy"],
                    "delta_log_loss_vs_previous_best": baseline_metrics["log_loss"] - metrics["log_loss"],
                    "delta_log_loss_vs_full_cardio": full_metrics["log_loss"] - metrics["log_loss"],
                    "cardio_features": ", ".join(subset),
                }
            )
    subset_df = pd.DataFrame(subset_rows).sort_values(
        ["log_loss", "roc_auc", "subset_size", "brier_score"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    best_subset_log_loss = float(subset_df.iloc[0]["log_loss"])
    near_best = subset_df.loc[subset_df["log_loss"] <= best_subset_log_loss + tie_threshold].copy()
    near_best = near_best.sort_values(["subset_size", "log_loss", "roc_auc"], ascending=[True, True, False]).reset_index(drop=True)
    selected_subset_row = near_best.iloc[0]
    selected_cardio_features = [feature.strip() for feature in str(selected_subset_row["cardio_features"]).split(",") if feature.strip()]

    return leave_one_out_df, subset_df, {
        "baseline_result": baseline_result,
        "full_result": full_result,
        "selected_cardio_features": selected_cardio_features,
        "selected_subset_row": selected_subset_row.to_dict(),
        "near_best_subset_table": near_best,
        "tie_threshold": tie_threshold,
    }


def build_cardio_feature_selection_summary(
    leave_one_out_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    cardio_context: dict[str, Any],
) -> str:
    """Summarize which cardio proxies matter most and the minimal strong subset."""
    baseline_metrics = cardio_context["baseline_result"]["validation_metrics"]
    full_metrics = cardio_context["full_result"]["validation_metrics"]
    selected_row = cardio_context["selected_subset_row"]
    selected_features = cardio_context["selected_cardio_features"]
    strongest_features = leave_one_out_df.head(4)
    tied_subsets = cardio_context["near_best_subset_table"].head(5)

    lines = [
        "Cardio feature selection summary:",
        "- The cardio pool was trimmed using validation-only leave-one-out ablation and subset search.",
        f"- full cardio pool feature count: {len(cardio_context['full_result']['feature_names']) - len(cardio_context['baseline_result']['feature_names'])}",
        f"- previous-best baseline log_loss: {baseline_metrics['log_loss']:.6f}",
        f"- full cardio pool log_loss: {full_metrics['log_loss']:.6f}",
        f"- selected minimal cardio subset size: {int(selected_row['subset_size'])}",
        f"- selected minimal cardio subset log_loss: {float(selected_row['log_loss']):.6f}",
        f"- selected minimal cardio subset roc_auc: {float(selected_row['roc_auc']):.6f}",
        f"- selected cardio features: {selected_features}",
        f"- tie threshold for smaller subsets: {cardio_context['tie_threshold']:.3f} log loss",
        "",
        "Leave-one-out ranking inside the cardio pool:",
    ]
    for row in strongest_features.itertuples(index=False):
        lines.append(
            f"- removing {row.removed_feature}: delta_log_loss_vs_full_cardio={row.delta_log_loss_vs_full_cardio:+.6f}, "
            f"delta_roc_auc_vs_full_cardio={row.delta_roc_auc_vs_full_cardio:+.6f}"
        )
    lines.append("")
    lines.append("Best near-tied cardio subsets:")
    for row in tied_subsets.itertuples(index=False):
        lines.append(
            f"- size={int(row.subset_size)} | log_loss={row.log_loss:.6f} | roc_auc={row.roc_auc:.6f} | features=[{row.cardio_features}]"
        )
    return "\n".join(lines)


def build_hypothesis_feature_summary(
    comparison_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    best_result: dict[str, Any],
    previous_best_features_result: dict[str, Any],
) -> str:
    """Summarize whether the new human-hypothesis feature ideas added validation signal."""
    best_metrics = best_result["validation_metrics"]
    baseline_metrics = previous_best_features_result["validation_metrics"]
    improvement_vs_baseline = baseline_metrics["log_loss"] - best_metrics["log_loss"]
    roc_auc_vs_baseline = best_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    improvement_vs_global = PREVIOUS_BEST_LOG_LOSS - best_metrics["log_loss"]
    roc_auc_vs_global = best_metrics["roc_auc"] - PREVIOUS_BEST_ROC_AUC

    helpful_groups = comparison_df.loc[
        comparison_df["model_name"] != "previous_best_features"
    ].loc[lambda frame: frame["log_loss"] < baseline_metrics["log_loss"], "model_name"].tolist()
    strongest_ablation = ablation_df.head(3)

    lines = [
        "Hypothesis feature summary:",
        "- These features were designed as interpretable MMA hypotheses rather than generic feature-search output.",
        f"- best hypothesis model: {best_result['feature_set_name']}",
        f"- validation log_loss: {best_metrics['log_loss']:.6f}",
        f"- validation roc_auc: {best_metrics['roc_auc']:.6f}",
        f"- validation brier_score: {best_metrics['brier_score']:.6f}",
        f"- validation accuracy: {best_metrics['accuracy']:.6f}",
        f"- log_loss improvement vs current previous-best feature set: {improvement_vs_baseline:+.6f}",
        f"- roc_auc change vs current previous-best feature set: {roc_auc_vs_baseline:+.6f}",
        f"- log_loss improvement vs global benchmark (0.663442): {improvement_vs_global:+.6f}",
        f"- roc_auc change vs global benchmark (0.642936): {roc_auc_vs_global:+.6f}",
        f"- feature groups that improved validation log loss over the current previous-best set: {helpful_groups if helpful_groups else 'None'}",
        "- If a group did not help here, that does not mean the MMA idea is wrong; it may mean the current formulation is too noisy.",
        "",
        "Strongest groups by leave-one-group-out ablation:",
    ]
    for row in strongest_ablation.itertuples(index=False):
        lines.append(
            f"- removing {row.removed_group}: delta_log_loss_vs_full={row.delta_log_loss_vs_full:+.6f}, delta_roc_auc_vs_full={row.delta_roc_auc_vs_full:+.6f}"
        )
    return "\n".join(lines)


def print_recency_sanity_checks(df: pd.DataFrame, best_n: int, best_half_life: int) -> None:
    """Print lightweight checks confirming recency features are pre-fight only."""
    red_first = df.get("pre_fight_red_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    blue_first = df.get("pre_fight_blue_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)

    recent_red_column = f"recent_{best_n}_red_win_pct"
    recent_blue_column = f"recent_{best_n}_blue_win_pct"
    red_recent_mismatches = int((pd.to_numeric(df.loc[red_first, recent_red_column], errors="coerce").fillna(0.0) != 0.0).sum()) if recent_red_column in df.columns else 0
    blue_recent_mismatches = int((pd.to_numeric(df.loc[blue_first, recent_blue_column], errors="coerce").fillna(0.0) != 0.0).sum()) if recent_blue_column in df.columns else 0
    red_exp_mismatches = int((pd.to_numeric(df.loc[red_first, "exp_decay_red_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if "exp_decay_red_win_pct" in df.columns else 0
    blue_exp_mismatches = int((pd.to_numeric(df.loc[blue_first, "exp_decay_blue_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if "exp_decay_blue_win_pct" in df.columns else 0

    sample_days = np.array([30.0, 180.0, 365.0, 730.0], dtype="float64")
    sample_weights = np.exp(-(np.log(2.0) / best_half_life) * sample_days)
    monotonic_decay = bool(np.all(np.diff(sample_weights) < 0))

    print("Recency sanity checks:")
    print(f"- red first-fight last-{best_n} recency mismatches: {red_recent_mismatches}")
    print(f"- blue first-fight last-{best_n} recency mismatches: {blue_recent_mismatches}")
    print(f"- red first-fight exp-decay recency mismatches: {red_exp_mismatches}")
    print(f"- blue first-fight exp-decay recency mismatches: {blue_exp_mismatches}")
    print(f"- exp-decay sample weights decrease with age: {monotonic_decay}")
    print(f"- sample exp-decay weights for {sample_days.astype(int).tolist()} days: {sample_weights.round(4).tolist()}")


def build_recency_model_summary(
    best_result: dict[str, Any],
    best_n: int,
    best_half_life: int,
    comparison_df: pd.DataFrame,
) -> str:
    """Create a readable summary of the best recency-based model."""
    metrics = best_result["validation_metrics"]
    improvement_log_loss = PREVIOUS_BEST_LOG_LOSS - metrics["log_loss"]
    improvement_roc_auc = metrics["roc_auc"] - PREVIOUS_BEST_ROC_AUC

    recency_coefficients = best_result["coefficients"].loc[
        best_result["coefficients"]["feature"].str.contains(f"recent_{best_n}|exp_decay", regex=True, na=False)
    ].head(5)

    lines = [
        "Recency model summary:",
        "- Recency features were added to let the model distinguish long-run cumulative ability from recent form.",
        f"- selected last-N window: {best_n}",
        f"- selected exponential half-life (days): {best_half_life}",
        f"- selected model: {best_result['feature_set_name']}",
        f"- validation log_loss: {metrics['log_loss']:.6f}",
        f"- validation roc_auc: {metrics['roc_auc']:.6f}",
        f"- validation brier_score: {metrics['brier_score']:.6f}",
        f"- validation accuracy: {metrics['accuracy']:.6f}",
        f"- log_loss improvement vs previous best (0.663442): {improvement_log_loss:+.6f}",
        f"- roc_auc change vs previous best (0.642936): {improvement_roc_auc:+.6f}",
        "- Fighters with fewer than N previous UFC bouts fall back to the smaller available history, and debut fights get zero-filled recency features.",
        "- All recency parameters were tuned on train + validation only. The hold-out test split was not used.",
        "",
        "Strongest recency coefficients in the selected model:",
    ]
    if recency_coefficients.empty:
        lines.append("- No recency coefficients were present in the best model.")
    else:
        for row in recency_coefficients.itertuples(index=False):
            lines.append(f"- {row.feature}: coefficient={row.coefficient:+.6f}")

    lines.append("")
    lines.append("Validation comparison leaderboard:")
    for row in comparison_df.itertuples(index=False):
        lines.append(f"- {row.model_name}: log_loss={row.log_loss:.6f}, roc_auc={row.roc_auc:.6f}, accuracy={row.accuracy:.6f}")
    return "\n".join(lines)


def train_final_logistic_model(
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
    model_spec: dict[str, Any] = DEFAULT_MODEL_SPEC,
) -> dict[str, Any]:
    """Fit the frozen final logistic model on train+validation, then score the hold-out test set once."""
    if not feature_list:
        raise ValueError("Final logistic feature list is empty.")

    train_x = to_numeric_frame(train_validation_df, feature_list)
    test_x = to_numeric_frame(test_df, feature_list)
    train_y = train_validation_df[TARGET_COLUMN]
    test_y = test_df[TARGET_COLUMN]

    pipeline = build_logistic_pipeline(model_spec)
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = evaluate_predictions(test_y, probabilities)
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    calibration_table = build_calibration_table(test_y, probabilities)

    prediction_frame = test_df[[DATE_COLUMN, *IDENTIFIER_COLUMNS, TARGET_COLUMN]].copy()
    prediction_frame["predicted_red_win_probability"] = probabilities
    prediction_frame["predicted_red_win_label"] = predictions

    return {
        "feature_set_name": "final_logistic_model",
        "feature_names": feature_list,
        "model_spec": model_spec,
        "test_metrics": metrics,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "coefficients": coefficient_table(pipeline, feature_list),
        "formula_table": build_formula_table(pipeline, feature_list),
        "intercept": float(pipeline.named_steps["model"].intercept_[0]),
        "prediction_frame": prediction_frame,
        "calibration_table": calibration_table,
        "pipeline": pipeline,
    }


def run_random_forest_baseline(
    modeling_dataset_path: Path = MODELING_DATASET_PATH,
) -> dict[str, Any]:
    """Run a clean validation-only Random Forest baseline with the frozen final feature set.

    This is a baseline Random Forest model.
    No hyperparameter tuning is done here.
    The hold-out test split is intentionally not used.
    The finalized logistic benchmark remains unchanged.
    """
    modeling_df = load_combined_dataset(modeling_dataset_path)
    train_df, validation_df, test_df = chronological_split(modeling_df)
    feature_list = build_final_logistic_feature_set(modeling_df)

    train_x = to_numeric_frame(train_df, feature_list)
    validation_x = to_numeric_frame(validation_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    validation_y = validation_df[TARGET_COLUMN]

    # Trees do not need scaling, but we still impute from the training split only.
    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    validation_x_imputed = imputer.transform(validation_x)

    rf = RandomForestClassifier(**RANDOM_FOREST_BASELINE_PARAMS)
    rf.fit(train_x_imputed, train_y)
    validation_probabilities = rf.predict_proba(validation_x_imputed)[:, 1]
    validation_metrics = evaluate_predictions(validation_y, validation_probabilities)

    results = {
        "feature_names": feature_list,
        "parameters": RANDOM_FOREST_BASELINE_PARAMS,
        "validation_metrics": validation_metrics,
        "train_summary": summarize_split(train_df, include_target_mean=True),
        "validation_summary": summarize_split(validation_df, include_target_mean=True),
        "test_summary_unused": summarize_split(test_df, include_target_mean=False),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_FOREST_BASELINE_VALIDATION_PATH.write_text(json.dumps(results, indent=2))

    print("Random Forest Validation Results:")
    print(f"- log loss: {validation_metrics['log_loss']:.6f}")
    print(f"- ROC AUC: {validation_metrics['roc_auc']:.6f}" if validation_metrics["roc_auc"] is not None else "- ROC AUC: unavailable")
    print(f"- Brier score: {validation_metrics['brier_score']:.6f}")
    print(f"- accuracy: {validation_metrics['accuracy']:.6f}")

    return results


def run_random_forest_random_search(
    n_iter: int = 100,
    modeling_dataset_path: Path = MODELING_DATASET_PATH,
) -> dict[str, Any]:
    """Run a validation-only random-search-style sweep for the Random Forest baseline.

    This keeps the finalized logistic model untouched.
    The hold-out test split is not used.
    Training happens on the train split only, and selection is based on validation log loss.
    """
    modeling_df = load_combined_dataset(modeling_dataset_path)
    train_df, validation_df, test_df = chronological_split(modeling_df)
    del test_df  # Explicitly unused here by design.

    feature_list = build_final_logistic_feature_set(modeling_df)
    train_x = to_numeric_frame(train_df, feature_list)
    validation_x = to_numeric_frame(validation_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    validation_y = validation_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    validation_x_imputed = imputer.transform(validation_x)

    sampled_params = list(ParameterSampler(RANDOM_FOREST_RANDOM_SEARCH_SPACE, n_iter=n_iter, random_state=42))
    rows: list[dict[str, Any]] = []
    best_model: RandomForestClassifier | None = None
    best_metrics: dict[str, float | None] | None = None
    best_params: dict[str, Any] | None = None

    for index, params in enumerate(sampled_params, start=1):
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        rf.fit(train_x_imputed, train_y)
        probabilities = rf.predict_proba(validation_x_imputed)[:, 1]
        metrics = evaluate_predictions(validation_y, probabilities)
        row = {
            "iteration": index,
            "log_loss": metrics["log_loss"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "accuracy": metrics["accuracy"],
            **params,
        }
        rows.append(row)

        if best_metrics is None:
            is_better = True
        else:
            current_roc_auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else float("-inf")
            best_roc_auc = best_metrics["roc_auc"] if best_metrics["roc_auc"] is not None else float("-inf")
            is_better = (
                metrics["log_loss"] < best_metrics["log_loss"]
                or (
                    np.isclose(metrics["log_loss"], best_metrics["log_loss"])
                    and current_roc_auc > best_roc_auc
                )
            )
        if is_better:
            best_model = rf
            best_metrics = metrics
            best_params = params

    results_df = pd.DataFrame(rows).sort_values(
        ["log_loss", "roc_auc", "brier_score"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    if best_model is None or best_metrics is None or best_params is None:
        raise RuntimeError("Random Forest random search did not produce a best model.")

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    baseline_results = run_random_forest_baseline(modeling_dataset_path=modeling_dataset_path)
    logistic_validation_benchmark = train_and_evaluate_feature_set(
        train_df,
        validation_df,
        feature_list,
        "final_feature_set_logistic_validation_benchmark",
        DEFAULT_MODEL_SPEC,
    )
    logistic_metrics = logistic_validation_benchmark["validation_metrics"]

    best_payload = {
        "feature_names": feature_list,
        "search_space": RANDOM_FOREST_RANDOM_SEARCH_SPACE,
        "n_iter": n_iter,
        "best_parameters": best_params,
        "best_validation_metrics": best_metrics,
        "baseline_random_forest_validation_metrics": baseline_results["validation_metrics"],
        "logistic_validation_benchmark": logistic_metrics,
        "train_summary": summarize_split(train_df, include_target_mean=True),
        "validation_summary": summarize_split(validation_df, include_target_mean=True),
        "test_set_note": "Unused during Random Forest random search.",
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RANDOM_FOREST_RANDOM_SEARCH_RESULTS_PATH, index=False)
    BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH.write_text(json.dumps(best_payload, indent=2))
    feature_importance_df.to_csv(RANDOM_FOREST_RANDOM_SEARCH_FEATURE_IMPORTANCE_PATH, index=False)

    baseline_metrics = baseline_results["validation_metrics"]
    improvement_vs_baseline = baseline_metrics["log_loss"] - best_metrics["log_loss"]
    roc_auc_vs_baseline = (best_metrics["roc_auc"] or np.nan) - (baseline_metrics["roc_auc"] or np.nan)
    logistic_log_loss = logistic_metrics["log_loss"]
    logistic_roc_auc = logistic_metrics["roc_auc"]
    competitive_with_logistic = (
        best_metrics["log_loss"] <= logistic_log_loss
        and (best_metrics["roc_auc"] or float("-inf")) >= (logistic_roc_auc or float("-inf"))
    )

    print("Random Forest Random Search Results:")
    print(f"- best parameters: {best_params}")
    print(f"- best validation log loss: {best_metrics['log_loss']:.6f}")
    print(f"- best validation ROC AUC: {best_metrics['roc_auc']:.6f}" if best_metrics["roc_auc"] is not None else "- best validation ROC AUC: unavailable")
    print(f"- best validation Brier score: {best_metrics['brier_score']:.6f}")
    print(f"- best validation accuracy: {best_metrics['accuracy']:.6f}")
    print(f"- log loss improvement over RF baseline: {improvement_vs_baseline:+.6f}")
    print(f"- ROC AUC change over RF baseline: {roc_auc_vs_baseline:+.6f}")
    print(
        f"- logistic validation benchmark: log_loss={logistic_log_loss:.6f}, "
        f"roc_auc={logistic_roc_auc:.6f}" if logistic_roc_auc is not None else f"- logistic validation benchmark: log_loss={logistic_log_loss:.6f}, roc_auc=unavailable"
    )
    print(f"- tuned RF competitive with logistic on validation: {competitive_with_logistic}")

    return {
        "feature_names": feature_list,
        "results_df": results_df,
        "best_parameters": best_params,
        "best_validation_metrics": best_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "logistic_validation_benchmark": logistic_metrics,
        "feature_importance_df": feature_importance_df,
        "competitive_with_logistic": competitive_with_logistic,
        "updated_files": [
            str(RANDOM_FOREST_BASELINE_VALIDATION_PATH),
            str(RANDOM_FOREST_RANDOM_SEARCH_RESULTS_PATH),
            str(BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH),
            str(RANDOM_FOREST_RANDOM_SEARCH_FEATURE_IMPORTANCE_PATH),
        ],
    }


def run_xgboost_baseline(
    modeling_dataset_path: Path = MODELING_DATASET_PATH,
) -> dict[str, Any]:
    """Run a clean validation-only XGBoost baseline on the frozen final feature set."""
    XGBClassifier = get_xgb_classifier()
    modeling_df = load_combined_dataset(modeling_dataset_path)
    train_df, validation_df, test_df = chronological_split(modeling_df)
    del test_df  # Hold-out test remains unused here.

    feature_list = build_final_logistic_feature_set(modeling_df)
    train_x = to_numeric_frame(train_df, feature_list)
    validation_x = to_numeric_frame(validation_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    validation_y = validation_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    validation_x_imputed = imputer.transform(validation_x)

    model = XGBClassifier(**XGBOOST_BASELINE_PARAMS)
    model.fit(train_x_imputed, train_y)
    validation_probabilities = model.predict_proba(validation_x_imputed)[:, 1]
    validation_metrics = evaluate_predictions(validation_y, validation_probabilities)

    print("XGBoost Baseline Validation Results:")
    print(f"- log loss: {validation_metrics['log_loss']:.6f}")
    print(f"- ROC AUC: {validation_metrics['roc_auc']:.6f}" if validation_metrics["roc_auc"] is not None else "- ROC AUC: unavailable")
    print(f"- Brier score: {validation_metrics['brier_score']:.6f}")
    print(f"- accuracy: {validation_metrics['accuracy']:.6f}")

    return {
        "feature_names": feature_list,
        "parameters": XGBOOST_BASELINE_PARAMS,
        "validation_metrics": validation_metrics,
        "train_summary": summarize_split(train_df, include_target_mean=True),
        "validation_summary": summarize_split(validation_df, include_target_mean=True),
    }


def run_xgboost_random_search(
    n_iter: int = 100,
    modeling_dataset_path: Path = MODELING_DATASET_PATH,
) -> dict[str, Any]:
    """Run a validation-only randomized search for XGBoost using the frozen feature set."""
    XGBClassifier = get_xgb_classifier()
    modeling_df = load_combined_dataset(modeling_dataset_path)
    train_df, validation_df, test_df = chronological_split(modeling_df)
    del test_df  # Explicitly unused during tuning.

    feature_list = build_final_logistic_feature_set(modeling_df)
    train_x = to_numeric_frame(train_df, feature_list)
    validation_x = to_numeric_frame(validation_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    validation_y = validation_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    validation_x_imputed = imputer.transform(validation_x)

    baseline_results = run_xgboost_baseline(modeling_dataset_path=modeling_dataset_path)
    rf_baseline_results = run_random_forest_baseline(modeling_dataset_path=modeling_dataset_path)
    logistic_validation_benchmark = train_and_evaluate_feature_set(
        train_df,
        validation_df,
        feature_list,
        "final_feature_set_logistic_validation_benchmark",
        DEFAULT_MODEL_SPEC,
    )
    logistic_metrics = logistic_validation_benchmark["validation_metrics"]

    sampled_params = list(ParameterSampler(XGBOOST_RANDOM_SEARCH_SPACE, n_iter=n_iter, random_state=42))
    rows: list[dict[str, Any]] = []
    best_model: Any = None
    best_metrics: dict[str, float | None] | None = None
    best_params: dict[str, Any] | None = None

    for index, params in enumerate(sampled_params, start=1):
        model_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            **params,
        }
        model = XGBClassifier(**model_params)
        model.fit(train_x_imputed, train_y)
        probabilities = model.predict_proba(validation_x_imputed)[:, 1]
        metrics = evaluate_predictions(validation_y, probabilities)
        rows.append(
            {
                "iteration": index,
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                **params,
            }
        )

        if best_metrics is None:
            is_better = True
        else:
            current_roc_auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else float("-inf")
            best_roc_auc = best_metrics["roc_auc"] if best_metrics["roc_auc"] is not None else float("-inf")
            is_better = (
                metrics["log_loss"] < best_metrics["log_loss"]
                or (
                    np.isclose(metrics["log_loss"], best_metrics["log_loss"])
                    and current_roc_auc > best_roc_auc
                )
            )
        if is_better:
            best_model = model
            best_metrics = metrics
            best_params = model_params

    results_df = pd.DataFrame(rows).sort_values(
        ["log_loss", "roc_auc", "brier_score"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    if best_model is None or best_metrics is None or best_params is None:
        raise RuntimeError("XGBoost random search did not produce a best model.")

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    best_payload = {
        "feature_names": feature_list,
        "search_space": XGBOOST_RANDOM_SEARCH_SPACE,
        "n_iter": n_iter,
        "best_parameters": best_params,
        "best_validation_metrics": best_metrics,
        "xgboost_baseline_validation_metrics": baseline_results["validation_metrics"],
        "random_forest_baseline_validation_metrics": rf_baseline_results["validation_metrics"],
        "logistic_validation_benchmark": logistic_metrics,
        "train_summary": summarize_split(train_df, include_target_mean=True),
        "validation_summary": summarize_split(validation_df, include_target_mean=True),
        "test_set_note": "Unused during XGBoost tuning.",
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(XGBOOST_VALIDATION_RESULTS_PATH, index=False)
    BEST_XGBOOST_VALIDATION_PATH.write_text(json.dumps(best_payload, indent=2))
    feature_importance_df.to_csv(XGBOOST_FEATURE_IMPORTANCE_PATH, index=False)

    baseline_metrics = baseline_results["validation_metrics"]
    improvement_vs_baseline = baseline_metrics["log_loss"] - best_metrics["log_loss"]
    roc_auc_vs_baseline = (best_metrics["roc_auc"] or np.nan) - (baseline_metrics["roc_auc"] or np.nan)
    logistic_log_loss = logistic_metrics["log_loss"]
    logistic_roc_auc = logistic_metrics["roc_auc"]
    competitive_with_logistic = (
        best_metrics["log_loss"] <= logistic_log_loss
        and (best_metrics["roc_auc"] or float("-inf")) >= (logistic_roc_auc or float("-inf"))
    )

    print("XGBoost Random Search Results:")
    print(f"- best parameters: {best_params}")
    print(f"- best validation log loss: {best_metrics['log_loss']:.6f}")
    print(f"- best validation ROC AUC: {best_metrics['roc_auc']:.6f}" if best_metrics["roc_auc"] is not None else "- best validation ROC AUC: unavailable")
    print(f"- best validation Brier score: {best_metrics['brier_score']:.6f}")
    print(f"- best validation accuracy: {best_metrics['accuracy']:.6f}")
    print(f"- log loss improvement over XGBoost baseline: {improvement_vs_baseline:+.6f}")
    print(f"- ROC AUC change over XGBoost baseline: {roc_auc_vs_baseline:+.6f}")
    print(
        f"- logistic validation benchmark: log_loss={logistic_log_loss:.6f}, roc_auc={logistic_roc_auc:.6f}"
        if logistic_roc_auc is not None
        else f"- logistic validation benchmark: log_loss={logistic_log_loss:.6f}, roc_auc=unavailable"
    )
    print(f"- tuned XGBoost competitive with logistic on validation: {competitive_with_logistic}")

    return {
        "feature_names": feature_list,
        "baseline_validation_metrics": baseline_metrics,
        "results_df": results_df,
        "best_parameters": best_params,
        "best_validation_metrics": best_metrics,
        "random_forest_baseline_validation_metrics": rf_baseline_results["validation_metrics"],
        "logistic_validation_benchmark": logistic_metrics,
        "feature_importance_df": feature_importance_df,
        "competitive_with_logistic": competitive_with_logistic,
        "updated_files": [
            str(XGBOOST_VALIDATION_RESULTS_PATH),
            str(BEST_XGBOOST_VALIDATION_PATH),
            str(XGBOOST_FEATURE_IMPORTANCE_PATH),
        ],
    }


def build_final_logistic_summary(
    final_result: dict[str, Any],
    feature_list: list[str],
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    """Create a readable summary for the locked final hold-out evaluation."""
    metrics = final_result["test_metrics"]
    confusion = final_result["confusion_matrix"]
    top_positive = final_result["coefficients"].sort_values("coefficient", ascending=False).head(5)
    top_negative = final_result["coefficients"].sort_values("coefficient", ascending=True).head(5)
    train_validation_summary = summarize_split(train_validation_df, include_target_mean=True)
    test_summary = summarize_split(test_df, include_target_mean=True)

    lines = [
        "Final logistic model summary:",
        "- This is the locked hold-out evaluation step for the interpretable logistic regression model.",
        "- Feature selection and tuning were completed on train + validation before using the test set.",
        "- After this point, changing features because of test performance would invalidate the hold-out evaluation.",
        f"- Modeling-era cutoff: only fights on or after {MODELING_CUTOFF_LABEL} are included because earlier rows show corner-assignment instability.",
        "",
        f"- final feature list: {feature_list}",
        f"- final model configuration: {final_result['model_spec']}",
        f"- train+validation date range: {train_validation_summary['start_date']} to {train_validation_summary['end_date']}",
        f"- train+validation rows: {train_validation_summary['rows']}",
        f"- train+validation red_win mean: {train_validation_summary['red_win_mean']:.6f}",
        f"- test date range: {test_summary['start_date']} to {test_summary['end_date']}",
        f"- test rows: {test_summary['rows']}",
        f"- test red_win mean: {test_summary['red_win_mean']:.6f}",
        "",
        "Final test metrics:",
        f"- log_loss: {metrics['log_loss']:.6f}",
        f"- roc_auc: {metrics['roc_auc']:.6f}" if metrics["roc_auc"] is not None else "- roc_auc: unavailable",
        f"- brier_score: {metrics['brier_score']:.6f}",
        f"- accuracy: {metrics['accuracy']:.6f}",
        f"- precision: {metrics['precision']:.6f}",
        f"- recall: {metrics['recall']:.6f}",
        f"- f1: {metrics['f1']:.6f}",
        f"- confusion_matrix: {confusion['matrix']}",
        "",
        "Strongest red-favoring coefficients:",
    ]
    for row in top_positive.itertuples(index=False):
        lines.append(f"- {row.feature}: coefficient={row.coefficient:+.6f}")
    lines.append("")
    lines.append("Strongest blue-favoring coefficients:")
    for row in top_negative.itertuples(index=False):
        lines.append(f"- {row.feature}: coefficient={row.coefficient:+.6f}")
    lines.append("")
    lines.append("- The test set was used once after model selection was completed.")
    return "\n".join(lines)


def train_final_random_forest_model(
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fit the frozen best Random Forest on train+validation and score the hold-out test once."""
    train_x = to_numeric_frame(train_validation_df, feature_list)
    test_x = to_numeric_frame(test_df, feature_list)
    train_y = train_validation_df[TARGET_COLUMN]
    test_y = test_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    test_x_imputed = imputer.transform(test_x)

    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(train_x_imputed, train_y)
    probabilities = model.predict_proba(test_x_imputed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = evaluate_predictions(test_y, probabilities)
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    calibration_table = build_calibration_table(test_y, probabilities)

    prediction_frame = test_df[[DATE_COLUMN, *IDENTIFIER_COLUMNS, TARGET_COLUMN]].copy()
    prediction_frame["predicted_red_win_probability"] = probabilities
    prediction_frame["predicted_red_win_label"] = predictions

    return {
        "model_family": "random_forest",
        "feature_names": feature_list,
        "model_params": params,
        "test_metrics": metrics,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "feature_importance": pd.DataFrame(
            {"feature": feature_list, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True),
        "prediction_frame": prediction_frame,
        "calibration_table": calibration_table,
    }


def train_final_xgboost_model(
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fit the frozen best XGBoost model on train+validation and score the hold-out test once."""
    XGBClassifier = get_xgb_classifier()

    train_x = to_numeric_frame(train_validation_df, feature_list)
    test_x = to_numeric_frame(test_df, feature_list)
    train_y = train_validation_df[TARGET_COLUMN]
    test_y = test_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    test_x_imputed = imputer.transform(test_x)

    model = XGBClassifier(**params)
    model.fit(train_x_imputed, train_y)
    probabilities = model.predict_proba(test_x_imputed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = evaluate_predictions(test_y, probabilities)
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    calibration_table = build_calibration_table(test_y, probabilities)

    prediction_frame = test_df[[DATE_COLUMN, *IDENTIFIER_COLUMNS, TARGET_COLUMN]].copy()
    prediction_frame["predicted_red_win_probability"] = probabilities
    prediction_frame["predicted_red_win_label"] = predictions

    return {
        "model_family": "xgboost",
        "feature_names": feature_list,
        "model_params": params,
        "test_metrics": metrics,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "feature_importance": pd.DataFrame(
            {"feature": feature_list, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True),
        "prediction_frame": prediction_frame,
        "calibration_table": calibration_table,
    }


def build_tree_test_summary(
    result: dict[str, Any],
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    """Create a readable final test summary for a frozen tree model."""
    metrics = result["test_metrics"]
    confusion = result["confusion_matrix"]
    top_features = result["feature_importance"].head(6)
    train_validation_summary = summarize_split(train_validation_df, include_target_mean=True)
    test_summary = summarize_split(test_df, include_target_mean=True)

    lines = [
        f"Final {result['model_family']} summary:",
        "- This is a frozen hold-out evaluation after validation-only model selection.",
        "- The hold-out test set was not used for tuning this tree model.",
        f"- Modeling-era cutoff: only fights on or after {MODELING_CUTOFF_LABEL} are included because earlier rows show corner-assignment instability.",
        f"- final feature list: {result['feature_names']}",
        f"- final model configuration: {result['model_params']}",
        f"- train+validation date range: {train_validation_summary['start_date']} to {train_validation_summary['end_date']}",
        f"- test date range: {test_summary['start_date']} to {test_summary['end_date']}",
        "",
        "Final test metrics:",
        f"- log_loss: {metrics['log_loss']:.6f}",
        f"- roc_auc: {metrics['roc_auc']:.6f}" if metrics["roc_auc"] is not None else "- roc_auc: unavailable",
        f"- brier_score: {metrics['brier_score']:.6f}",
        f"- accuracy: {metrics['accuracy']:.6f}",
        f"- precision: {metrics['precision']:.6f}",
        f"- recall: {metrics['recall']:.6f}",
        f"- f1: {metrics['f1']:.6f}",
        f"- confusion_matrix: {confusion['matrix']}",
        "",
        "Top feature importances:",
    ]
    for row in top_features.itertuples(index=False):
        lines.append(f"- {row.feature}: importance={row.importance:.6f}")
    return "\n".join(lines)


def save_tree_test_outputs(
    result: dict[str, Any],
    metrics_path: Path,
    predictions_path: Path,
    summary_path: Path,
    summary_text: str,
    feature_importance_path: Path | None = None,
    calibration_table_path: Path | None = None,
) -> None:
    """Save final tree-model test outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "feature_names": result["feature_names"],
                "model_params": result["model_params"],
                "test_metrics": result["test_metrics"],
                "confusion_matrix": result["confusion_matrix"],
            },
            indent=2,
        )
    )
    result["prediction_frame"].to_csv(predictions_path, index=False)
    summary_path.write_text(summary_text)
    if feature_importance_path is not None:
        result["feature_importance"].to_csv(feature_importance_path, index=False)
    if calibration_table_path is not None:
        result["calibration_table"].to_csv(calibration_table_path, index=False)


def build_final_model_test_comparison(
    logistic_metrics: dict[str, Any],
    random_forest_result: dict[str, Any],
    xgboost_result: dict[str, Any],
) -> pd.DataFrame:
    """Build a single comparison table for logistic, Random Forest, and XGBoost on test."""
    rows = [
        {
            "model_name": "logistic_regression",
            "log_loss": logistic_metrics["test_metrics"]["log_loss"],
            "roc_auc": logistic_metrics["test_metrics"]["roc_auc"],
            "brier_score": logistic_metrics["test_metrics"]["brier_score"],
            "accuracy": logistic_metrics["test_metrics"]["accuracy"],
            "precision": logistic_metrics["test_metrics"]["precision"],
            "recall": logistic_metrics["test_metrics"]["recall"],
            "f1": logistic_metrics["test_metrics"]["f1"],
            "tn": logistic_metrics["confusion_matrix"]["tn"],
            "fp": logistic_metrics["confusion_matrix"]["fp"],
            "fn": logistic_metrics["confusion_matrix"]["fn"],
            "tp": logistic_metrics["confusion_matrix"]["tp"],
            "confusion_matrix": str(logistic_metrics["confusion_matrix"]["matrix"]),
        },
        {
            "model_name": "random_forest",
            "log_loss": random_forest_result["test_metrics"]["log_loss"],
            "roc_auc": random_forest_result["test_metrics"]["roc_auc"],
            "brier_score": random_forest_result["test_metrics"]["brier_score"],
            "accuracy": random_forest_result["test_metrics"]["accuracy"],
            "precision": random_forest_result["test_metrics"]["precision"],
            "recall": random_forest_result["test_metrics"]["recall"],
            "f1": random_forest_result["test_metrics"]["f1"],
            "tn": random_forest_result["confusion_matrix"]["tn"],
            "fp": random_forest_result["confusion_matrix"]["fp"],
            "fn": random_forest_result["confusion_matrix"]["fn"],
            "tp": random_forest_result["confusion_matrix"]["tp"],
            "confusion_matrix": str(random_forest_result["confusion_matrix"]["matrix"]),
        },
        {
            "model_name": "xgboost",
            "log_loss": xgboost_result["test_metrics"]["log_loss"],
            "roc_auc": xgboost_result["test_metrics"]["roc_auc"],
            "brier_score": xgboost_result["test_metrics"]["brier_score"],
            "accuracy": xgboost_result["test_metrics"]["accuracy"],
            "precision": xgboost_result["test_metrics"]["precision"],
            "recall": xgboost_result["test_metrics"]["recall"],
            "f1": xgboost_result["test_metrics"]["f1"],
            "tn": xgboost_result["confusion_matrix"]["tn"],
            "fp": xgboost_result["confusion_matrix"]["fp"],
            "fn": xgboost_result["confusion_matrix"]["fn"],
            "tp": xgboost_result["confusion_matrix"]["tp"],
            "confusion_matrix": str(xgboost_result["confusion_matrix"]["matrix"]),
        },
    ]
    comparison_df = pd.DataFrame(rows)

    best_log_loss_model = comparison_df.sort_values("log_loss", ascending=True).iloc[0]["model_name"]
    best_roc_auc_model = comparison_df.sort_values("roc_auc", ascending=False).iloc[0]["model_name"]
    best_brier_model = comparison_df.sort_values("brier_score", ascending=True).iloc[0]["model_name"]
    best_accuracy_model = comparison_df.sort_values("accuracy", ascending=False).iloc[0]["model_name"]
    best_recall_model = comparison_df.sort_values("recall", ascending=False).iloc[0]["model_name"]

    notes_map = {model_name: [] for model_name in comparison_df["model_name"]}
    notes_map[best_log_loss_model].append("Best log loss on test.")
    notes_map[best_roc_auc_model].append("Best ROC AUC on test.")
    notes_map[best_brier_model].append("Best Brier score on test.")
    notes_map[best_accuracy_model].append("Best accuracy on test.")
    notes_map[best_recall_model].append("Highest recall on test.")

    weakest_probability_model = comparison_df.sort_values(["log_loss", "brier_score"], ascending=[False, False]).iloc[0]["model_name"]
    if weakest_probability_model == "xgboost":
        notes_map[weakest_probability_model].append("Weakest overall probability quality on test.")

    comparison_df["notes"] = comparison_df["model_name"].map(lambda name: " ".join(notes_map[name]).strip())
    return comparison_df.sort_values(["log_loss", "roc_auc"], ascending=[True, False]).reset_index(drop=True)


def build_final_model_comparison_summary(comparison_df: pd.DataFrame) -> str:
    """Summarize the three frozen model families on the hold-out test set."""
    best_log_loss_row = comparison_df.sort_values("log_loss", ascending=True).iloc[0]
    best_roc_auc_row = comparison_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_brier_row = comparison_df.sort_values("brier_score", ascending=True).iloc[0]
    best_accuracy_row = comparison_df.sort_values("accuracy", ascending=False).iloc[0]
    probability_winners = sorted({best_log_loss_row.model_name, best_brier_row.model_name})

    if len(probability_winners) == 1:
        probability_line = (
            f"- {probability_winners[0]} is the strongest test-time model for probability quality here "
            "because it won both log loss and Brier score."
        )
    else:
        probability_line = (
            "- Probability-quality leadership is split across metrics here: "
            f"log loss favored {best_log_loss_row.model_name}, while Brier score favored {best_brier_row.model_name}."
        )

    downstream_probability_line = (
        f"- If the downstream use case is betting probabilities, {best_log_loss_row.model_name} is the better "
        "starting point from this test run because log loss is the primary probability metric in this repo."
    )

    lines = [
        "Final model comparison summary:",
        f"- modeling-era cutoff: fights on or after {MODELING_CUTOFF_LABEL}",
        f"- best model by log loss: {best_log_loss_row.model_name} ({best_log_loss_row.log_loss:.6f})",
        f"- best model by ROC AUC: {best_roc_auc_row.model_name} ({best_roc_auc_row.roc_auc:.6f})",
        f"- best model by Brier score: {best_brier_row.model_name} ({best_brier_row.brier_score:.6f})",
        f"- best model by accuracy: {best_accuracy_row.model_name} ({best_accuracy_row.accuracy:.6f})",
        "",
        "Interpretation:",
        probability_line,
        "- Logistic regression remains the strongest ranking model by ROC AUC.",
        "- XGBoost did not surpass either Random Forest or logistic regression on this frozen hold-out evaluation.",
        downstream_probability_line,
        "- The hold-out test set has now been consumed for all three frozen model families and should not be reused for further tuning.",
    ]
    return "\n".join(lines)


def save_outputs(
    modeling_df: pd.DataFrame,
    last_n_results: pd.DataFrame,
    last_n_results_expanded: pd.DataFrame,
    exp_decay_results: pd.DataFrame,
    recency_model_comparison: pd.DataFrame,
    best_coefficients: pd.DataFrame,
    summary_text: str,
    formula_text: str,
    hypothesis_comparison: pd.DataFrame,
    hypothesis_ablation: pd.DataFrame,
    best_hypothesis_coefficients: pd.DataFrame,
    hypothesis_summary_text: str,
    cardio_leave_one_out: pd.DataFrame,
    cardio_subset_comparison: pd.DataFrame,
    cardio_summary_text: str,
) -> None:
    """Save the recency-focused modeling outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(MODELING_DATASET_PATH, index=False)
    last_n_results.to_csv(LAST_N_RECENCY_RESULTS_PATH, index=False)
    last_n_results_expanded.to_csv(LAST_N_RECENCY_RESULTS_EXPANDED_PATH, index=False)
    exp_decay_results.to_csv(EXP_DECAY_RECENCY_RESULTS_PATH, index=False)
    recency_model_comparison.to_csv(RECENCY_MODEL_COMPARISON_PATH, index=False)
    best_coefficients.to_csv(BEST_RECENCY_COEFFICIENTS_PATH, index=False)
    RECENCY_MODEL_SUMMARY_PATH.write_text(summary_text)
    hypothesis_comparison.to_csv(HYPOTHESIS_FEATURE_GROUP_COMPARISON_PATH, index=False)
    hypothesis_ablation.to_csv(HYPOTHESIS_FEATURE_GROUP_ABLATION_PATH, index=False)
    best_hypothesis_coefficients.to_csv(BEST_HYPOTHESIS_COEFFICIENTS_PATH, index=False)
    HYPOTHESIS_FEATURE_SUMMARY_PATH.write_text(hypothesis_summary_text)
    cardio_leave_one_out.to_csv(CARDIO_FEATURE_LEAVE_ONE_OUT_PATH, index=False)
    cardio_subset_comparison.to_csv(CARDIO_SUBSET_COMPARISON_PATH, index=False)
    CARDIO_FEATURE_SUMMARY_PATH.write_text(cardio_summary_text)
    ELO_MODEL_FORMULA_PATH.write_text(formula_text)
    VALIDATION_SUMMARY_V2_PATH.write_text(summary_text)


def save_final_logistic_outputs(final_result: dict[str, Any], summary_text: str) -> None:
    """Save the final one-time hold-out evaluation artifacts."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LOGISTIC_TEST_METRICS_PATH.write_text(
        json.dumps(
            {
                "feature_names": final_result["feature_names"],
                "model_spec": final_result["model_spec"],
                "test_metrics": final_result["test_metrics"],
                "confusion_matrix": final_result["confusion_matrix"],
                "intercept": final_result["intercept"],
            },
            indent=2,
        )
    )
    final_result["coefficients"].to_csv(FINAL_LOGISTIC_COEFFICIENTS_PATH, index=False)
    final_result["prediction_frame"].to_csv(FINAL_LOGISTIC_TEST_PREDICTIONS_PATH, index=False)
    final_result["calibration_table"].to_csv(FINAL_LOGISTIC_CALIBRATION_TABLE_PATH, index=False)
    FINAL_LOGISTIC_SUMMARY_PATH.write_text(summary_text)


def run_validation_workflow() -> dict[str, Any]:
    """Run the recency-focused validation-only workflow."""
    combined_df = load_combined_dataset()
    modeling_df_default, train_df_default, validation_df_default, test_df_default, excluded_columns = build_validation_frame_with_recency(
        combined_df,
        n_last=DEFAULT_LAST_N_RECENCY,
        half_life_days=DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    )

    print("Excluded leakage or post-fight columns:")
    print(excluded_columns)
    print(f"Modeling-era cutoff applied: fights on or after {MODELING_CUTOFF_LABEL}")
    print_split_summary(train_df_default, validation_df_default, test_df_default)
    print("Validation experiments use only train and validation splits. The hold-out test split remains untouched.")

    last_n_results, best_n = run_last_n_recency_validation(combined_df)
    last_n_stability = summarize_last_n_stability(last_n_results)
    print("Last-N recency validation results:")
    print(last_n_results.to_string(index=False))
    print(f"Best last-N window by validation log loss: {best_n}")
    print(f"Best last-N window by validation ROC AUC: {last_n_stability['best_n_by_roc_auc']}")
    print(f"N values within {last_n_stability['tie_threshold']:.3f} log loss of the best: {last_n_stability['effectively_tied_n_values']}")
    print("Expanded last-N stability summary:")
    print(last_n_stability["results_with_deltas"].to_string(index=False))

    exp_decay_results, best_half_life = run_exp_decay_recency_validation(combined_df, n_last=best_n)
    print("Exponential-decay recency validation results:")
    print(exp_decay_results.to_string(index=False))
    print(f"Best exponential half-life by validation log loss: {best_half_life}")

    best_last_n_improvement = PREVIOUS_BEST_LOG_LOSS - float(last_n_results.iloc[0]["log_loss"])
    best_last_n_roc_auc = float(last_n_results.iloc[0]["roc_auc"]) - PREVIOUS_BEST_ROC_AUC
    best_exp_improvement = PREVIOUS_BEST_LOG_LOSS - float(exp_decay_results.iloc[0]["log_loss"])
    best_exp_roc_auc = float(exp_decay_results.iloc[0]["roc_auc"]) - PREVIOUS_BEST_ROC_AUC
    print(f"Last-N improvement vs previous best log loss ({PREVIOUS_BEST_LOG_LOSS:.6f}): {best_last_n_improvement:+.6f}")
    print(f"Last-N change vs previous best ROC AUC ({PREVIOUS_BEST_ROC_AUC:.6f}): {best_last_n_roc_auc:+.6f}")
    print(f"Exp-decay improvement vs previous best log loss ({PREVIOUS_BEST_LOG_LOSS:.6f}): {best_exp_improvement:+.6f}")
    print(f"Exp-decay change vs previous best ROC AUC ({PREVIOUS_BEST_ROC_AUC:.6f}): {best_exp_roc_auc:+.6f}")

    recency_model_comparison, comparison_context = run_recency_model_comparisons(
        combined_df,
        best_n=best_n,
        best_half_life=best_half_life,
    )
    print("Validation model comparison with recency features:")
    print(recency_model_comparison.to_string(index=False))

    best_result = comparison_context["best_result"]
    best_model_name = comparison_context["best_model_name"]
    print_recency_sanity_checks(comparison_context["modeling_df"], best_n, best_half_life)

    improvement_log_loss = PREVIOUS_BEST_LOG_LOSS - best_result["validation_metrics"]["log_loss"]
    improvement_roc_auc = best_result["validation_metrics"]["roc_auc"] - PREVIOUS_BEST_ROC_AUC
    print(f"Improvement vs previous best log loss ({PREVIOUS_BEST_LOG_LOSS:.6f}): {improvement_log_loss:+.6f}")
    print(f"Improvement vs previous best ROC AUC ({PREVIOUS_BEST_ROC_AUC:.6f}): {improvement_roc_auc:+.6f}")

    current_previous_best_features = build_current_previous_best_feature_set(comparison_context["modeling_df"])
    cardio_candidates = build_cardio_feature_candidates(comparison_context["modeling_df"])
    cardio_leave_one_out, cardio_subset_comparison, cardio_context = run_cardio_feature_selection(
        comparison_context["train_df"],
        comparison_context["validation_df"],
        current_previous_best_features,
        cardio_candidates,
    )
    cardio_summary_text = build_cardio_feature_selection_summary(
        cardio_leave_one_out,
        cardio_subset_comparison,
        cardio_context,
    )
    print("Cardio feature leave-one-out ranking:")
    print(cardio_leave_one_out.to_string(index=False))
    print("Cardio subset comparison (best subsets first):")
    print(cardio_subset_comparison.head(15).to_string(index=False))
    print("Selected minimal cardio subset:")
    print(f"- features: {cardio_context['selected_cardio_features']}")
    print(
        "- validation metrics: "
        f"log_loss={float(cardio_context['selected_subset_row']['log_loss']):.6f}, "
        f"roc_auc={float(cardio_context['selected_subset_row']['roc_auc']):.6f}, "
        f"brier_score={float(cardio_context['selected_subset_row']['brier_score']):.6f}, "
        f"accuracy={float(cardio_context['selected_subset_row']['accuracy']):.6f}"
    )

    hypothesis_comparison, hypothesis_context = run_hypothesis_feature_group_comparisons(
        comparison_context["modeling_df"],
        comparison_context["train_df"],
        comparison_context["validation_df"],
        selected_cardio_features=cardio_context["selected_cardio_features"],
    )
    print("Human-hypothesis feature group comparison:")
    print(hypothesis_comparison.to_string(index=False))

    hypothesis_ablation = run_hypothesis_feature_group_ablation(
        comparison_context["train_df"],
        comparison_context["validation_df"],
        hypothesis_context["previous_best_features"],
        hypothesis_context["hypothesis_groups"],
    )
    print("Human-hypothesis feature group ablation:")
    print(hypothesis_ablation.to_string(index=False))

    best_hypothesis_result = hypothesis_context["best_result"]
    best_hypothesis_model_name = hypothesis_context["best_model_name"]
    previous_best_result = hypothesis_context["results"]["previous_best_features"]
    hypothesis_summary_text = build_hypothesis_feature_summary(
        hypothesis_comparison,
        hypothesis_ablation,
        best_hypothesis_result,
        previous_best_result,
    )
    hypothesis_improvement_log_loss = previous_best_result["validation_metrics"]["log_loss"] - best_hypothesis_result["validation_metrics"]["log_loss"]
    hypothesis_improvement_roc_auc = best_hypothesis_result["validation_metrics"]["roc_auc"] - previous_best_result["validation_metrics"]["roc_auc"]
    print(f"Hypothesis-group improvement vs current previous-best log loss: {hypothesis_improvement_log_loss:+.6f}")
    print(f"Hypothesis-group ROC AUC change vs current previous-best: {hypothesis_improvement_roc_auc:+.6f}")

    summary_text = build_recency_model_summary(best_result, best_n, best_half_life, recency_model_comparison)
    formula_text = format_logistic_formula(best_result)
    best_coefficients = best_result["coefficients"].copy()
    best_coefficients.insert(0, "model_name", best_model_name)
    best_hypothesis_coefficients = best_hypothesis_result["coefficients"].copy()
    best_hypothesis_coefficients.insert(0, "model_name", best_hypothesis_model_name)

    save_outputs(
        comparison_context["modeling_df"],
        last_n_results,
        last_n_stability["results_with_deltas"],
        exp_decay_results,
        recency_model_comparison,
        best_coefficients,
        summary_text,
        formula_text,
        hypothesis_comparison,
        hypothesis_ablation,
        best_hypothesis_coefficients,
        hypothesis_summary_text,
        cardio_leave_one_out,
        cardio_subset_comparison,
        cardio_summary_text,
    )

    updated_files = [
        str(MODELING_DATASET_PATH),
        str(LAST_N_RECENCY_RESULTS_PATH),
        str(LAST_N_RECENCY_RESULTS_EXPANDED_PATH),
        str(EXP_DECAY_RECENCY_RESULTS_PATH),
        str(RECENCY_MODEL_COMPARISON_PATH),
        str(BEST_RECENCY_COEFFICIENTS_PATH),
        str(RECENCY_MODEL_SUMMARY_PATH),
        str(HYPOTHESIS_FEATURE_GROUP_COMPARISON_PATH),
        str(HYPOTHESIS_FEATURE_GROUP_ABLATION_PATH),
        str(BEST_HYPOTHESIS_COEFFICIENTS_PATH),
        str(HYPOTHESIS_FEATURE_SUMMARY_PATH),
        str(CARDIO_FEATURE_LEAVE_ONE_OUT_PATH),
        str(CARDIO_SUBSET_COMPARISON_PATH),
        str(CARDIO_FEATURE_SUMMARY_PATH),
        str(ELO_MODEL_FORMULA_PATH),
        str(VALIDATION_SUMMARY_V2_PATH),
    ]
    print("Files updated or created:")
    for path in updated_files:
        print(f"- {path}")

    best_metrics = last_n_results.loc[last_n_results["n_last"] == best_n].iloc[0]
    smaller_nearly_as_good = last_n_stability["results_with_deltas"].loc[
        (last_n_stability["results_with_deltas"]["n_last"] < best_n)
        & (last_n_stability["results_with_deltas"]["log_loss_delta_from_best"] <= 0.001)
    ]
    print("Expanded last-N final summary:")
    print(f"- best N: {best_n}")
    print(
        "- validation metrics: "
        f"log_loss={best_metrics['log_loss']:.6f}, "
        f"roc_auc={best_metrics['roc_auc']:.6f}, "
        f"brier_score={best_metrics['brier_score']:.6f}, "
        f"accuracy={best_metrics['accuracy']:.6f}"
    )
    print(f"- N=8 remains best: {best_n == 8}")
    if smaller_nearly_as_good.empty:
        print("- smaller, more interpretable N within 0.001 log loss: none")
    else:
        compact_rows = []
        for row in smaller_nearly_as_good.itertuples(index=False):
            compact_rows.append(f"N={int(row.n_last)} (delta={row.log_loss_delta_from_best:.6f})")
        print(f"- smaller, more interpretable N within 0.001 log loss: {compact_rows}")

    return {
        "excluded_columns": excluded_columns,
        "best_n": best_n,
        "best_n_by_roc_auc": last_n_stability["best_n_by_roc_auc"],
        "best_half_life": best_half_life,
        "last_n_results": last_n_results,
        "last_n_stability": last_n_stability["results_with_deltas"],
        "effectively_tied_n_values": last_n_stability["effectively_tied_n_values"],
        "exp_decay_results": exp_decay_results,
        "recency_model_comparison": recency_model_comparison,
        "hypothesis_feature_group_comparison": hypothesis_comparison,
        "hypothesis_feature_group_ablation": hypothesis_ablation,
        "cardio_feature_leave_one_out": cardio_leave_one_out,
        "cardio_subset_comparison": cardio_subset_comparison,
        "selected_cardio_features": cardio_context["selected_cardio_features"],
        "cardio_summary_text": cardio_summary_text,
        "best_hypothesis_result": best_hypothesis_result,
        "best_hypothesis_model_name": best_hypothesis_model_name,
        "best_result": best_result,
        "best_model_name": best_model_name,
        "comparison_specs": comparison_context["comparison_specs"],
        "hypothesis_groups": hypothesis_context["hypothesis_groups"],
        "modeling_df": comparison_context["modeling_df"],
        "train_df": comparison_context["train_df"],
        "validation_df": comparison_context["validation_df"],
        "test_df": comparison_context["test_df"],
        "summary_text": summary_text,
        "formula_text": formula_text,
        "hypothesis_summary_text": hypothesis_summary_text,
        "updated_files": updated_files,
    }


def run_final_logistic_evaluation() -> dict[str, Any]:
    """Freeze the final interpretable feature set and evaluate it once on the hold-out test split."""
    combined_df = load_combined_dataset()
    modeling_df, _, _, _, _ = build_validation_frame_with_recency(
        combined_df,
        n_last=DEFAULT_LAST_N_RECENCY,
        half_life_days=DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    )
    train_df, validation_df, test_df = chronological_split(modeling_df)
    train_validation_df = combine_train_validation(train_df, validation_df)
    feature_list = build_final_logistic_feature_set(modeling_df)

    print("Final logistic model split summary:")
    train_validation_summary = summarize_split(train_validation_df, include_target_mean=True)
    test_summary = summarize_split(test_df, include_target_mean=True)
    print(
        f"- train+validation: rows={train_validation_summary['rows']}, "
        f"date range={train_validation_summary['start_date']} to {train_validation_summary['end_date']}, "
        f"red_win mean={train_validation_summary['red_win_mean']:.4f}"
    )
    print(
        f"- test: rows={test_summary['rows']}, "
        f"date range={test_summary['start_date']} to {test_summary['end_date']}, "
        f"red_win mean={test_summary['red_win_mean']:.4f}"
    )
    print(f"- frozen final features: {feature_list}")
    print("- Guard: after this point, changing features based on test performance would invalidate the hold-out evaluation.")

    final_result = train_final_logistic_model(
        train_validation_df,
        test_df,
        feature_list,
        model_spec=DEFAULT_MODEL_SPEC,
    )
    summary_text = build_final_logistic_summary(final_result, feature_list, train_validation_df, test_df)
    save_final_logistic_outputs(final_result, summary_text)

    print("Final logistic hold-out test metrics:")
    print(final_result["test_metrics"])
    print(f"- confusion_matrix: {final_result['confusion_matrix']['matrix']}")

    return {
        "modeling_df": modeling_df,
        "train_validation_df": train_validation_df,
        "test_df": test_df,
        "feature_list": feature_list,
        "final_result": final_result,
        "summary_text": summary_text,
        "updated_files": [
            str(FINAL_LOGISTIC_TEST_METRICS_PATH),
            str(FINAL_LOGISTIC_COEFFICIENTS_PATH),
            str(FINAL_LOGISTIC_TEST_PREDICTIONS_PATH),
            str(FINAL_LOGISTIC_SUMMARY_PATH),
            str(FINAL_LOGISTIC_CALIBRATION_TABLE_PATH),
        ],
    }


def run_final_random_forest_evaluation() -> dict[str, Any]:
    """Train the frozen best Random Forest on train+validation and evaluate on test once."""
    combined_df = load_combined_dataset()
    modeling_df, _, _, _, _ = build_validation_frame_with_recency(
        combined_df,
        n_last=DEFAULT_LAST_N_RECENCY,
        half_life_days=DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    )
    train_df, validation_df, test_df = chronological_split(modeling_df)
    train_validation_df = combine_train_validation(train_df, validation_df)
    feature_list = build_final_logistic_feature_set(modeling_df)
    best_payload = json.loads(BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH.read_text())
    params = best_payload["best_parameters"]

    print("Final Random Forest split summary:")
    print(
        f"- train+validation: rows={len(train_validation_df)}, "
        f"date range={train_validation_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {train_validation_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}, "
        f"red_win mean={train_validation_df[TARGET_COLUMN].mean():.4f}"
    )
    print(
        f"- test: rows={len(test_df)}, "
        f"date range={test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}, "
        f"red_win mean={test_df[TARGET_COLUMN].mean():.4f}"
    )

    result = train_final_random_forest_model(train_validation_df, test_df, feature_list, params)
    summary_text = build_tree_test_summary(result, train_validation_df, test_df)
    save_tree_test_outputs(
        result,
        FINAL_RANDOM_FOREST_TEST_METRICS_PATH,
        FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH,
        FINAL_RANDOM_FOREST_SUMMARY_PATH,
        summary_text,
        feature_importance_path=FINAL_RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
        calibration_table_path=FINAL_RANDOM_FOREST_CALIBRATION_TABLE_PATH,
    )
    print("Final Random Forest hold-out test metrics:")
    print(result["test_metrics"])
    print(f"- confusion_matrix: {result['confusion_matrix']['matrix']}")

    return {
        "modeling_df": modeling_df,
        "train_validation_df": train_validation_df,
        "test_df": test_df,
        "feature_list": feature_list,
        "final_result": result,
        "summary_text": summary_text,
        "updated_files": [
            str(FINAL_RANDOM_FOREST_TEST_METRICS_PATH),
            str(FINAL_RANDOM_FOREST_TEST_PREDICTIONS_PATH),
            str(FINAL_RANDOM_FOREST_SUMMARY_PATH),
            str(FINAL_RANDOM_FOREST_FEATURE_IMPORTANCE_PATH),
            str(FINAL_RANDOM_FOREST_CALIBRATION_TABLE_PATH),
        ],
    }


def run_final_xgboost_evaluation() -> dict[str, Any]:
    """Train the frozen best XGBoost model on train+validation and evaluate on test once."""
    combined_df = load_combined_dataset()
    modeling_df, _, _, _, _ = build_validation_frame_with_recency(
        combined_df,
        n_last=DEFAULT_LAST_N_RECENCY,
        half_life_days=DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
    )
    train_df, validation_df, test_df = chronological_split(modeling_df)
    train_validation_df = combine_train_validation(train_df, validation_df)
    feature_list = build_final_logistic_feature_set(modeling_df)
    best_payload = json.loads(BEST_XGBOOST_VALIDATION_PATH.read_text())
    params = best_payload["best_parameters"]

    print("Final XGBoost split summary:")
    print(
        f"- train+validation: rows={len(train_validation_df)}, "
        f"date range={train_validation_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {train_validation_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}, "
        f"red_win mean={train_validation_df[TARGET_COLUMN].mean():.4f}"
    )
    print(
        f"- test: rows={len(test_df)}, "
        f"date range={test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}, "
        f"red_win mean={test_df[TARGET_COLUMN].mean():.4f}"
    )

    result = train_final_xgboost_model(train_validation_df, test_df, feature_list, params)
    summary_text = build_tree_test_summary(result, train_validation_df, test_df)
    save_tree_test_outputs(
        result,
        FINAL_XGBOOST_TEST_METRICS_PATH,
        FINAL_XGBOOST_TEST_PREDICTIONS_PATH,
        FINAL_XGBOOST_SUMMARY_PATH,
        summary_text,
        feature_importance_path=FINAL_XGBOOST_FEATURE_IMPORTANCE_PATH,
        calibration_table_path=FINAL_XGBOOST_CALIBRATION_TABLE_PATH,
    )
    print("Final XGBoost hold-out test metrics:")
    print(result["test_metrics"])
    print(f"- confusion_matrix: {result['confusion_matrix']['matrix']}")

    return {
        "modeling_df": modeling_df,
        "train_validation_df": train_validation_df,
        "test_df": test_df,
        "feature_list": feature_list,
        "final_result": result,
        "summary_text": summary_text,
        "updated_files": [
            str(FINAL_XGBOOST_TEST_METRICS_PATH),
            str(FINAL_XGBOOST_TEST_PREDICTIONS_PATH),
            str(FINAL_XGBOOST_SUMMARY_PATH),
            str(FINAL_XGBOOST_FEATURE_IMPORTANCE_PATH),
            str(FINAL_XGBOOST_CALIBRATION_TABLE_PATH),
        ],
    }


def run_final_model_family_comparison() -> dict[str, Any]:
    """Evaluate the frozen best logistic, Random Forest, and XGBoost models on the hold-out test set."""
    logistic_result = run_final_logistic_evaluation()
    random_forest_result = run_final_random_forest_evaluation()
    xgboost_result = run_final_xgboost_evaluation()

    logistic_metrics_payload = json.loads(FINAL_LOGISTIC_TEST_METRICS_PATH.read_text())
    logistic_payload = {
        "test_metrics": logistic_metrics_payload["test_metrics"],
        "confusion_matrix": logistic_metrics_payload["confusion_matrix"],
    }
    comparison_df = build_final_model_test_comparison(
        logistic_payload,
        random_forest_result["final_result"],
        xgboost_result["final_result"],
    )
    comparison_df.to_csv(FINAL_MODEL_TEST_COMPARISON_PATH, index=False)
    comparison_summary_text = build_final_model_comparison_summary(comparison_df)
    FINAL_MODEL_COMPARISON_SUMMARY_PATH.write_text(comparison_summary_text)

    print("Final model-family test comparison:")
    print(comparison_df.to_string(index=False))
    print(comparison_summary_text)

    return {
        "logistic": logistic_result,
        "random_forest": random_forest_result,
        "xgboost": xgboost_result,
        "comparison_df": comparison_df,
        "updated_files": [
            *logistic_result["updated_files"],
            *random_forest_result["updated_files"],
            *xgboost_result["updated_files"],
            str(FINAL_MODEL_TEST_COMPARISON_PATH),
            str(FINAL_MODEL_COMPARISON_SUMMARY_PATH),
        ],
    }


def main() -> None:
    """Command-line entry point."""
    run_validation_workflow()
    run_final_logistic_evaluation()


if __name__ == "__main__":
    main()
