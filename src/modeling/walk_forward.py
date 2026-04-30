"""Walk-forward probability research with calibration and market benchmarking.

This module adds a separate research workflow that does not alter the frozen
final hold-out benchmark outputs. It uses the merged odds dataset, stays within
pre-test data, and evaluates out-of-sample probability quality across annual
walk-forward folds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src._pipeline_impl import (
    BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH,
    DATE_COLUMN,
    FINAL_LOGISTIC_FEATURE_CANDIDATES,
    MODELING_CUTOFF_DATE,
    MODELING_CUTOFF_LABEL,
    MODELING_DATASET_WITH_ODDS_PATH,
    OUTPUTS_DIR,
    TARGET_COLUMN,
    build_calibration_table,
    build_final_logistic_feature_set,
    chronological_split,
    combine_train_validation,
    evaluate_predictions,
    summarize_split,
    to_numeric_frame,
)

WALK_FORWARD_FOLD_SUMMARY_PATH = OUTPUTS_DIR / "walk_forward_fold_summary.csv"
WALK_FORWARD_METHOD_RESULTS_BY_FOLD_PATH = OUTPUTS_DIR / "walk_forward_method_results_by_fold.csv"
WALK_FORWARD_MODEL_COMPARISON_PATH = OUTPUTS_DIR / "walk_forward_model_comparison.csv"
WALK_FORWARD_PREDICTION_PANEL_PATH = OUTPUTS_DIR / "walk_forward_prediction_panel.csv"
WALK_FORWARD_CALIBRATION_COMPARISON_PATH = OUTPUTS_DIR / "walk_forward_calibration_comparison.csv"
WALK_FORWARD_YEARLY_METRICS_PATH = OUTPUTS_DIR / "walk_forward_yearly_metrics.csv"
WALK_FORWARD_BLEND_WEIGHTS_PATH = OUTPUTS_DIR / "walk_forward_blend_weights.csv"
WALK_FORWARD_MARKET_SUMMARY_PATH = OUTPUTS_DIR / "walk_forward_market_benchmark_summary.txt"

MODEL_PROBABILITY_COLUMNS = {
    "market_novig": "market_prob_red",
    "rf_raw": "rf_raw_prob",
    "rf_sigmoid": "rf_sigmoid_prob",
    "rf_isotonic": "rf_isotonic_prob",
    "rf_raw_market_blend": "rf_raw_market_blend_prob",
    "rf_sigmoid_market_blend": "rf_sigmoid_market_blend_prob",
    "rf_isotonic_market_blend": "rf_isotonic_market_blend_prob",
}

DEFAULT_BLEND_WEIGHT_GRID = tuple(np.round(np.linspace(0.0, 1.0, 11), 2).tolist())
DEFAULT_MIN_TRAIN_ROWS = 250
DEFAULT_MIN_CALIBRATION_ROWS = 50
DEFAULT_MIN_TEST_ROWS = 50


def _load_frozen_random_forest_params() -> dict[str, Any]:
    """Load the validation-selected Random Forest configuration."""
    if BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH.exists():
        payload = json.loads(BEST_RANDOM_FOREST_RANDOM_SEARCH_PATH.read_text())
        params = payload.get("best_parameters", {}).copy()
        if params:
            return params
    return {
        "n_estimators": 800,
        "max_depth": 12,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.3,
        "bootstrap": True,
        "criterion": "log_loss",
        "class_weight": "balanced_subsample",
    }


def _clip_probabilities(values: np.ndarray | pd.Series) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for stable transforms."""
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _logit(values: np.ndarray | pd.Series) -> np.ndarray:
    """Convert probabilities to logits after clipping."""
    clipped = _clip_probabilities(values)
    return np.log(clipped / (1.0 - clipped))


def load_walk_forward_research_dataframe(
    path: Path = MODELING_DATASET_WITH_ODDS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the merged odds dataset and keep only the pre-test research universe.

    The repo's final hold-out test has already been consumed for frozen benchmark
    reporting. This research workflow therefore stays inside the earlier
    train+validation region and builds walk-forward folds there.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Merged odds dataset not found at {path}. Run run_odds_ingestion_pipeline() first."
        )

    merged_df = pd.read_csv(path, parse_dates=[DATE_COLUMN], low_memory=False)
    merged_df = merged_df.loc[pd.to_datetime(merged_df[DATE_COLUMN], errors="coerce") >= MODELING_CUTOFF_DATE].copy()
    train_df, validation_df, test_df = chronological_split(merged_df)
    research_universe = combine_train_validation(train_df, validation_df)
    return merged_df, research_universe, train_df, validation_df


def prepare_walk_forward_market_dataset(
    path: Path = MODELING_DATASET_WITH_ODDS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the pre-test, odds-matched rows used in walk-forward research."""
    merged_df, research_universe, train_df, validation_df = load_walk_forward_research_dataframe(path=path)
    feature_list = build_final_logistic_feature_set(research_universe)

    required_columns = [
        DATE_COLUMN,
        "red_fighter_name",
        "blue_fighter_name",
        TARGET_COLUMN,
        "red_decimal_odds",
        "blue_decimal_odds",
        "red_implied_prob_novig",
        "blue_implied_prob_novig",
        *feature_list,
    ]
    missing_columns = [column for column in required_columns if column not in research_universe.columns]
    if missing_columns:
        raise KeyError(f"Walk-forward research dataset is missing required columns: {missing_columns}")

    valid_mask = research_universe[required_columns].notna().all(axis=1)
    valid_df = research_universe.loc[valid_mask].copy()
    valid_df = valid_df.sort_values([DATE_COLUMN, "red_fighter_name", "blue_fighter_name"], kind="mergesort").reset_index(drop=True)
    valid_df["event_year"] = valid_df[DATE_COLUMN].dt.year.astype(int)
    valid_df["market_prob_red"] = pd.to_numeric(valid_df["red_implied_prob_novig"], errors="coerce")
    valid_df["market_prob_blue"] = pd.to_numeric(valid_df["blue_implied_prob_novig"], errors="coerce")

    diagnostics = {
        "full_rows": int(len(merged_df)),
        "research_rows": int(len(research_universe)),
        "research_date_range": summarize_split(research_universe, include_target_mean=False),
        "research_train_rows": int(len(train_df)),
        "research_validation_rows": int(len(validation_df)),
        "matched_odds_rows": int(len(valid_df)),
        "matched_odds_rate_within_research": float(len(valid_df) / len(research_universe)) if len(research_universe) else 0.0,
        "feature_list": feature_list,
    }
    return valid_df, diagnostics


def build_annual_walk_forward_folds(
    df: pd.DataFrame,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    min_calibration_rows: int = DEFAULT_MIN_CALIBRATION_ROWS,
    min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
) -> list[dict[str, Any]]:
    """Build expanding annual train/calibration/test folds.

    Fold structure for test year Y:
    - training data: all years < Y - 1
    - calibration data: year Y - 1
    - test data: year Y
    """
    if df.empty:
        return []

    years = sorted(df["event_year"].dropna().unique().tolist())
    folds: list[dict[str, Any]] = []
    fold_id = 1
    for index in range(2, len(years)):
        calibration_year = years[index - 1]
        test_year = years[index]
        train_df = df.loc[df["event_year"] < calibration_year].copy()
        calibration_df = df.loc[df["event_year"] == calibration_year].copy()
        test_df = df.loc[df["event_year"] == test_year].copy()
        if len(train_df) < min_train_rows or len(calibration_df) < min_calibration_rows or len(test_df) < min_test_rows:
            continue
        folds.append(
            {
                "fold_id": fold_id,
                "train_years": sorted(train_df["event_year"].unique().tolist()),
                "calibration_year": int(calibration_year),
                "test_year": int(test_year),
                "train_df": train_df,
                "calibration_df": calibration_df,
                "test_df": test_df,
            }
        )
        fold_id += 1
    return folds


def _fit_sigmoid_calibrator(calibration_probabilities: np.ndarray, calibration_y: pd.Series) -> LogisticRegression | None:
    """Fit Platt-style calibration on raw model probabilities."""
    if pd.Series(calibration_y).nunique() < 2:
        return None
    x = _logit(calibration_probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(x, calibration_y)
    return calibrator


def _predict_sigmoid_calibrated(calibrator: LogisticRegression | None, probabilities: np.ndarray) -> np.ndarray:
    """Apply a fitted sigmoid calibrator, falling back to raw probabilities."""
    if calibrator is None:
        return _clip_probabilities(probabilities)
    x = _logit(probabilities).reshape(-1, 1)
    return _clip_probabilities(calibrator.predict_proba(x)[:, 1])


def _fit_isotonic_calibrator(calibration_probabilities: np.ndarray, calibration_y: pd.Series) -> IsotonicRegression | None:
    """Fit isotonic calibration on raw model probabilities."""
    if len(calibration_probabilities) == 0:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(_clip_probabilities(calibration_probabilities), calibration_y)
    return calibrator


def _predict_isotonic_calibrated(calibrator: IsotonicRegression | None, probabilities: np.ndarray) -> np.ndarray:
    """Apply a fitted isotonic calibrator, falling back to raw probabilities."""
    if calibrator is None:
        return _clip_probabilities(probabilities)
    return _clip_probabilities(calibrator.predict(_clip_probabilities(probabilities)))


def _choose_best_blend_weight(
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    y_true: pd.Series,
    weight_grid: tuple[float, ...] = DEFAULT_BLEND_WEIGHT_GRID,
) -> tuple[float, float]:
    """Choose the model/market blend weight on the calibration slice only."""
    rows: list[dict[str, float]] = []
    for weight in weight_grid:
        blended = _clip_probabilities(weight * model_probabilities + (1.0 - weight) * market_probabilities)
        metrics = evaluate_predictions(y_true, blended)
        rows.append({"weight": float(weight), "log_loss": float(metrics["log_loss"])})
    comparison = pd.DataFrame(rows).sort_values(["log_loss", "weight"], ascending=[True, False]).reset_index(drop=True)
    best_row = comparison.iloc[0]
    return float(best_row["weight"]), float(best_row["log_loss"])


def _fit_random_forest_fold_model(
    train_df: pd.DataFrame,
    feature_list: list[str],
    params: dict[str, Any],
) -> tuple[SimpleImputer, RandomForestClassifier]:
    """Fit one Random Forest model on the fold training set."""
    train_x = to_numeric_frame(train_df, feature_list)
    train_y = train_df[TARGET_COLUMN]
    imputer = SimpleImputer(strategy="median")
    train_x_imputed = imputer.fit_transform(train_x)
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(train_x_imputed, train_y)
    return imputer, model


def _predict_random_forest_fold(
    imputer: SimpleImputer,
    model: RandomForestClassifier,
    frame: pd.DataFrame,
    feature_list: list[str],
) -> np.ndarray:
    """Generate probability predictions for one fold slice."""
    x = to_numeric_frame(frame, feature_list)
    x_imputed = imputer.transform(x)
    return _clip_probabilities(model.predict_proba(x_imputed)[:, 1])


def _build_fold_prediction_frame(
    fold: dict[str, Any],
    test_df: pd.DataFrame,
    probability_columns: dict[str, np.ndarray],
    blend_weight_summary: dict[str, float],
) -> pd.DataFrame:
    """Return one fold's out-of-sample prediction panel."""
    panel = test_df[[DATE_COLUMN, "red_fighter_name", "blue_fighter_name", TARGET_COLUMN, "event_year"]].copy()
    panel["fold_id"] = int(fold["fold_id"])
    panel["calibration_year"] = int(fold["calibration_year"])
    panel["test_year"] = int(fold["test_year"])
    for column_name, values in probability_columns.items():
        panel[column_name] = values
    for column_name, value in blend_weight_summary.items():
        panel[column_name] = value
    return panel


def _evaluate_methods_by_fold(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each probability source within one fold."""
    rows: list[dict[str, Any]] = []
    for method_name, column_name in MODEL_PROBABILITY_COLUMNS.items():
        metrics = evaluate_predictions(panel[TARGET_COLUMN], panel[column_name].to_numpy())
        rows.append(
            {
                "fold_id": int(panel["fold_id"].iloc[0]),
                "calibration_year": int(panel["calibration_year"].iloc[0]),
                "test_year": int(panel["test_year"].iloc[0]),
                "model_name": method_name,
                "rows": int(len(panel)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_method_comparison(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate out-of-sample metrics across all walk-forward folds."""
    rows: list[dict[str, Any]] = []
    for method_name, column_name in MODEL_PROBABILITY_COLUMNS.items():
        probabilities = prediction_panel[column_name].to_numpy()
        metrics = evaluate_predictions(prediction_panel[TARGET_COLUMN], probabilities)
        calibration = build_calibration_table(prediction_panel[TARGET_COLUMN], probabilities)
        calibration_gap = float(
            (calibration["mean_predicted_probability"] - calibration["observed_red_win_rate"]).abs().mean()
        ) if not calibration.empty else np.nan
        rows.append(
            {
                "model_name": method_name,
                "rows": int(len(prediction_panel)),
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_probability": float(np.mean(probabilities)),
                "calibration_gap": calibration_gap,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["log_loss", "brier_score", "roc_auc"], ascending=[True, True, False]).reset_index(drop=True)
    market_log_loss = comparison.loc[comparison["model_name"].eq("market_novig"), "log_loss"].iloc[0]
    comparison["log_loss_vs_market"] = comparison["log_loss"] - market_log_loss
    return comparison


def _build_calibration_comparison(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    """Build decile calibration tables for each probability source."""
    tables: list[pd.DataFrame] = []
    for method_name, column_name in MODEL_PROBABILITY_COLUMNS.items():
        calibration = build_calibration_table(prediction_panel[TARGET_COLUMN], prediction_panel[column_name].to_numpy())
        calibration["model_name"] = method_name
        tables.append(calibration)
    if not tables:
        return pd.DataFrame()
    ordered_columns = [
        "model_name",
        "probability_bin",
        "rows",
        "mean_predicted_probability",
        "observed_red_win_rate",
    ]
    return pd.concat(tables, ignore_index=True)[ordered_columns]


def _build_yearly_metrics(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-year out-of-sample metrics for each method."""
    rows: list[dict[str, Any]] = []
    for test_year, year_df in prediction_panel.groupby("test_year", sort=True):
        for method_name, column_name in MODEL_PROBABILITY_COLUMNS.items():
            metrics = evaluate_predictions(year_df[TARGET_COLUMN], year_df[column_name].to_numpy())
            rows.append(
                {
                    "test_year": int(test_year),
                    "model_name": method_name,
                    "rows": int(len(year_df)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["test_year", "log_loss", "brier_score"], ascending=[True, True, True]).reset_index(drop=True)


def _build_summary_text(
    diagnostics: dict[str, Any],
    fold_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
) -> str:
    """Write a concise summary of the walk-forward research run."""
    best_model = comparison.iloc[0]
    market_row = comparison.loc[comparison["model_name"].eq("market_novig")].iloc[0]
    raw_row = comparison.loc[comparison["model_name"].eq("rf_raw")].iloc[0]
    sigmoid_row = comparison.loc[comparison["model_name"].eq("rf_sigmoid")].iloc[0]
    isotonic_row = comparison.loc[comparison["model_name"].eq("rf_isotonic")].iloc[0]
    blend_rows = comparison.loc[comparison["model_name"].str.contains("market_blend", na=False)]
    best_blend = blend_rows.iloc[0] if not blend_rows.empty else None

    lines = [
        "Walk-forward market benchmark summary:",
        "- This is a separate research workflow built on pre-test, odds-matched data only.",
        "- The consumed final hold-out benchmark outputs were not modified or reused for tuning here.",
        f"- frozen final feature set: {FINAL_LOGISTIC_FEATURE_CANDIDATES}",
        f"- modeling-era cutoff: fights on or after {MODELING_CUTOFF_LABEL}",
        f"- research rows before odds matching: {diagnostics['research_rows']}",
        f"- odds-matched research rows: {diagnostics['matched_odds_rows']}",
        f"- number of annual walk-forward folds: {len(fold_summary)}",
        "- fold design: expanding train, 1-year calibration, next-year test.",
        "",
        "Aggregate out-of-sample comparison:",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- {row.model_name}: log_loss={row.log_loss:.6f}, roc_auc={row.roc_auc:.6f}, "
            f"brier_score={row.brier_score:.6f}, accuracy={row.accuracy:.6f}, calibration_gap={row.calibration_gap:.6f}"
        )
    lines.extend(
        [
            "",
            f"- best method by log loss: {best_model['model_name']}",
            f"- market no-vig log_loss: {market_row['log_loss']:.6f}",
            f"- raw RF log_loss: {raw_row['log_loss']:.6f}",
            f"- sigmoid-calibrated RF log_loss: {sigmoid_row['log_loss']:.6f}",
            f"- isotonic-calibrated RF log_loss: {isotonic_row['log_loss']:.6f}",
            f"- calibration improved over raw RF by log loss: {min(sigmoid_row['log_loss'], isotonic_row['log_loss']) < raw_row['log_loss']}",
            f"- market beat raw RF by log loss: {market_row['log_loss'] < raw_row['log_loss']}",
            f"- best blend beat both market and raw RF by log loss: {best_blend is not None and best_blend['log_loss'] < min(market_row['log_loss'], raw_row['log_loss'])}",
        ]
    )
    if best_blend is not None:
        lines.append(
            f"- best blend method: {best_blend['model_name']} with log_loss={best_blend['log_loss']:.6f}"
        )
    if not yearly_metrics.empty:
        latest_year = int(yearly_metrics['test_year'].max())
        latest_year_best = yearly_metrics.loc[yearly_metrics['test_year'].eq(latest_year)].sort_values('log_loss').iloc[0]
        lines.append(
            f"- latest walk-forward year in this research sample: {latest_year}, best method that year: {latest_year_best['model_name']}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Market no-vig probabilities are the key external benchmark because they summarize tradable consensus information.",
            "- Calibration tests whether the model's probability scale is trustworthy, not just whether it can rank fights.",
            "- Blend tests ask whether the model adds incremental information on top of the market rather than replacing it outright.",
            "- Any future strategy rules should still be chosen without touching the consumed final hold-out test benchmark.",
        ]
    )
    return "\n".join(lines)


def run_walk_forward_market_research(
    modeling_dataset_with_odds_path: Path = MODELING_DATASET_WITH_ODDS_PATH,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    min_calibration_rows: int = DEFAULT_MIN_CALIBRATION_ROWS,
    min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
    blend_weight_grid: tuple[float, ...] = DEFAULT_BLEND_WEIGHT_GRID,
) -> dict[str, Any]:
    """Run the full walk-forward calibration and market benchmark workflow.

    This workflow is intentionally separate from the frozen final test benchmark.
    It uses only the earlier train+validation era from the odds-matched dataset.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    research_df, diagnostics = prepare_walk_forward_market_dataset(path=modeling_dataset_with_odds_path)
    folds = build_annual_walk_forward_folds(
        research_df,
        min_train_rows=min_train_rows,
        min_calibration_rows=min_calibration_rows,
        min_test_rows=min_test_rows,
    )
    if not folds:
        raise RuntimeError("No valid walk-forward folds were created. Check odds coverage and fold size thresholds.")

    feature_list = diagnostics["feature_list"]
    rf_params = _load_frozen_random_forest_params()

    prediction_panels: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, Any]] = []
    method_results_by_fold: list[pd.DataFrame] = []
    blend_weight_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_df = fold["train_df"]
        calibration_df = fold["calibration_df"]
        test_df = fold["test_df"]
        imputer, model = _fit_random_forest_fold_model(train_df, feature_list, rf_params)
        calibration_raw = _predict_random_forest_fold(imputer, model, calibration_df, feature_list)
        test_raw = _predict_random_forest_fold(imputer, model, test_df, feature_list)

        sigmoid_calibrator = _fit_sigmoid_calibrator(calibration_raw, calibration_df[TARGET_COLUMN])
        isotonic_calibrator = _fit_isotonic_calibrator(calibration_raw, calibration_df[TARGET_COLUMN])

        calibration_sigmoid = _predict_sigmoid_calibrated(sigmoid_calibrator, calibration_raw)
        test_sigmoid = _predict_sigmoid_calibrated(sigmoid_calibrator, test_raw)
        calibration_isotonic = _predict_isotonic_calibrated(isotonic_calibrator, calibration_raw)
        test_isotonic = _predict_isotonic_calibrated(isotonic_calibrator, test_raw)

        calibration_market = _clip_probabilities(calibration_df["market_prob_red"].to_numpy())
        test_market = _clip_probabilities(test_df["market_prob_red"].to_numpy())

        raw_blend_weight, raw_blend_calib_log_loss = _choose_best_blend_weight(
            calibration_raw,
            calibration_market,
            calibration_df[TARGET_COLUMN],
            weight_grid=blend_weight_grid,
        )
        sigmoid_blend_weight, sigmoid_blend_calib_log_loss = _choose_best_blend_weight(
            calibration_sigmoid,
            calibration_market,
            calibration_df[TARGET_COLUMN],
            weight_grid=blend_weight_grid,
        )
        isotonic_blend_weight, isotonic_blend_calib_log_loss = _choose_best_blend_weight(
            calibration_isotonic,
            calibration_market,
            calibration_df[TARGET_COLUMN],
            weight_grid=blend_weight_grid,
        )

        test_raw_blend = _clip_probabilities(raw_blend_weight * test_raw + (1.0 - raw_blend_weight) * test_market)
        test_sigmoid_blend = _clip_probabilities(sigmoid_blend_weight * test_sigmoid + (1.0 - sigmoid_blend_weight) * test_market)
        test_isotonic_blend = _clip_probabilities(isotonic_blend_weight * test_isotonic + (1.0 - isotonic_blend_weight) * test_market)

        probability_columns = {
            "market_prob_red": test_market,
            "rf_raw_prob": test_raw,
            "rf_sigmoid_prob": test_sigmoid,
            "rf_isotonic_prob": test_isotonic,
            "rf_raw_market_blend_prob": test_raw_blend,
            "rf_sigmoid_market_blend_prob": test_sigmoid_blend,
            "rf_isotonic_market_blend_prob": test_isotonic_blend,
        }
        blend_summary = {
            "raw_blend_weight": raw_blend_weight,
            "sigmoid_blend_weight": sigmoid_blend_weight,
            "isotonic_blend_weight": isotonic_blend_weight,
        }
        panel = _build_fold_prediction_frame(fold, test_df, probability_columns, blend_summary)
        prediction_panels.append(panel)
        method_results_by_fold.append(_evaluate_methods_by_fold(panel))

        fold_summary_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "train_start_date": train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "train_end_date": train_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                "calibration_start_date": calibration_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "calibration_end_date": calibration_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                "test_start_date": test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "test_end_date": test_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                "calibration_year": int(fold["calibration_year"]),
                "test_year": int(fold["test_year"]),
                "train_rows": int(len(train_df)),
                "calibration_rows": int(len(calibration_df)),
                "test_rows": int(len(test_df)),
                "train_red_win_mean": float(train_df[TARGET_COLUMN].mean()),
                "calibration_red_win_mean": float(calibration_df[TARGET_COLUMN].mean()),
                "test_red_win_mean": float(test_df[TARGET_COLUMN].mean()),
            }
        )
        blend_weight_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "calibration_year": int(fold["calibration_year"]),
                "test_year": int(fold["test_year"]),
                "raw_blend_weight": raw_blend_weight,
                "raw_blend_calibration_log_loss": raw_blend_calib_log_loss,
                "sigmoid_blend_weight": sigmoid_blend_weight,
                "sigmoid_blend_calibration_log_loss": sigmoid_blend_calib_log_loss,
                "isotonic_blend_weight": isotonic_blend_weight,
                "isotonic_blend_calibration_log_loss": isotonic_blend_calib_log_loss,
            }
        )

    prediction_panel = pd.concat(prediction_panels, ignore_index=True).sort_values([DATE_COLUMN, "fold_id"]).reset_index(drop=True)
    fold_summary = pd.DataFrame(fold_summary_rows).sort_values("fold_id").reset_index(drop=True)
    method_results = pd.concat(method_results_by_fold, ignore_index=True).sort_values(["fold_id", "log_loss", "brier_score"]).reset_index(drop=True)
    comparison = _aggregate_method_comparison(prediction_panel)
    calibration_comparison = _build_calibration_comparison(prediction_panel)
    yearly_metrics = _build_yearly_metrics(prediction_panel)
    blend_weights = pd.DataFrame(blend_weight_rows).sort_values("fold_id").reset_index(drop=True)
    summary_text = _build_summary_text(diagnostics, fold_summary, comparison, yearly_metrics)

    fold_summary.to_csv(WALK_FORWARD_FOLD_SUMMARY_PATH, index=False)
    method_results.to_csv(WALK_FORWARD_METHOD_RESULTS_BY_FOLD_PATH, index=False)
    comparison.to_csv(WALK_FORWARD_MODEL_COMPARISON_PATH, index=False)
    prediction_panel.to_csv(WALK_FORWARD_PREDICTION_PANEL_PATH, index=False)
    calibration_comparison.to_csv(WALK_FORWARD_CALIBRATION_COMPARISON_PATH, index=False)
    yearly_metrics.to_csv(WALK_FORWARD_YEARLY_METRICS_PATH, index=False)
    blend_weights.to_csv(WALK_FORWARD_BLEND_WEIGHTS_PATH, index=False)
    WALK_FORWARD_MARKET_SUMMARY_PATH.write_text(summary_text)

    best_row = comparison.iloc[0]
    market_row = comparison.loc[comparison["model_name"].eq("market_novig")].iloc[0]
    raw_row = comparison.loc[comparison["model_name"].eq("rf_raw")].iloc[0]

    print("Walk-forward market research summary:")
    print(f"- pre-test matched-odds rows: {diagnostics['matched_odds_rows']}")
    print(f"- annual folds: {len(fold_summary)}")
    print(f"- best method by log loss: {best_row['model_name']} ({best_row['log_loss']:.6f})")
    print(f"- market no-vig log loss: {market_row['log_loss']:.6f}")
    print(f"- raw RF log loss: {raw_row['log_loss']:.6f}")
    print(f"- calibration improved over raw RF: {(comparison.loc[comparison['model_name'].eq('rf_sigmoid'), 'log_loss'].iloc[0] < raw_row['log_loss']) or (comparison.loc[comparison['model_name'].eq('rf_isotonic'), 'log_loss'].iloc[0] < raw_row['log_loss'])}")
    print(f"- outputs saved: {WALK_FORWARD_MODEL_COMPARISON_PATH.name}, {WALK_FORWARD_MARKET_SUMMARY_PATH.name}")

    return {
        "diagnostics": diagnostics,
        "feature_list": feature_list,
        "rf_params": rf_params,
        "fold_summary": fold_summary,
        "method_results_by_fold": method_results,
        "comparison": comparison,
        "prediction_panel": prediction_panel,
        "calibration_comparison": calibration_comparison,
        "yearly_metrics": yearly_metrics,
        "blend_weights": blend_weights,
        "summary_text": summary_text,
        "updated_files": [
            str(WALK_FORWARD_FOLD_SUMMARY_PATH),
            str(WALK_FORWARD_METHOD_RESULTS_BY_FOLD_PATH),
            str(WALK_FORWARD_MODEL_COMPARISON_PATH),
            str(WALK_FORWARD_PREDICTION_PANEL_PATH),
            str(WALK_FORWARD_CALIBRATION_COMPARISON_PATH),
            str(WALK_FORWARD_YEARLY_METRICS_PATH),
            str(WALK_FORWARD_BLEND_WEIGHTS_PATH),
            str(WALK_FORWARD_MARKET_SUMMARY_PATH),
        ],
    }
