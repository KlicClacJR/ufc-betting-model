"""Build a cleaned UFC fight dataset with merged fighter and fight features."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGHT_PATH = PROJECT_ROOT / "data" / "ufc_fights_clean.csv"
DEFAULT_FIGHTER_STATISTICS_PATH = PROJECT_ROOT / "data" / "raw_fighter_statistics.csv"
# Backward-compatible alias for notebooks or imports that used the older name.
DEFAULT_FIGHTER_DETAILS_PATH = DEFAULT_FIGHTER_STATISTICS_PATH
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "combined_statistics.csv"

PLACEHOLDERS = {"", " ", "---", "--", "-", "null", "NULL", "None", "none", "nan", "NaN"}

STATIC_DETAIL_COLUMNS = {
    "fighter_name": ["fighter_name", "name", "fighter", "full_name", "fighter_full_name"],
    "height": ["height", "ht", "fighter_height", "height_in", "height_inches", "height_cm", "height_cms"],
    "weight": ["weight", "wt", "fighter_weight", "weight_lbs", "weight_in_lbs", "weight_kg", "weight_kgs"],
    "reach": ["reach", "fighter_reach", "reach_in", "reach_inches", "reach_cm", "reach_cms"],
    "stance": ["stance", "fighter_stance"],
    "dob": ["dob", "date_of_birth", "birth_date", "fighter_dob", "date_of_birth"],
}

FIGHTER_PROFILE_RATE_COLUMNS = {
    "slpm": ["slpm"],
    "str_acc": ["str_acc"],
    "sapm": ["sapm"],
    "str_def": ["str_def"],
    "td_avg": ["td_avg"],
    "td_acc": ["td_acc"],
    "td_def": ["td_def"],
    "sub_avg": ["sub_avg"],
}

FIGHTER_ATTRIBUTE_COLUMNS = [
    "red_fighter_height",
    "red_fighter_weight",
    "red_fighter_reach",
    "red_fighter_stance",
    "red_fighter_dob",
    "red_fighter_age",
    "blue_fighter_height",
    "blue_fighter_weight",
    "blue_fighter_reach",
    "blue_fighter_stance",
    "blue_fighter_dob",
    "blue_fighter_age",
    "red_fighter_slpm",
    "red_fighter_str_acc",
    "red_fighter_sapm",
    "red_fighter_str_def",
    "red_fighter_td_avg",
    "red_fighter_td_acc",
    "red_fighter_td_def",
    "red_fighter_sub_avg",
    "blue_fighter_slpm",
    "blue_fighter_str_acc",
    "blue_fighter_sapm",
    "blue_fighter_str_def",
    "blue_fighter_td_avg",
    "blue_fighter_td_acc",
    "blue_fighter_td_def",
    "blue_fighter_sub_avg",
]

ATTRIBUTE_DIAGNOSTIC_COLUMNS = [
    "red_fighter_height",
    "blue_fighter_height",
    "red_fighter_reach",
    "blue_fighter_reach",
    "red_fighter_dob",
    "blue_fighter_dob",
]

STAT_PAIR_SUFFIXES = [
    "sig_str",
    "total_str",
    "td",
    "sig_str_head",
    "sig_str_body",
    "sig_str_leg",
    "sig_str_distance",
    "sig_str_clinch",
    "sig_str_ground",
]

CUMULATIVE_SOURCE_COLUMNS = {
    "total_sig_strikes_landed": "fighter_sig_str_landed",
    "total_sig_strikes_attempted": "fighter_sig_str_attempted",
    "total_sig_strikes_head_landed": "fighter_sig_str_head_landed",
    "total_sig_strikes_head_attempted": "fighter_sig_str_head_attempted",
    "total_sig_strikes_body_landed": "fighter_sig_str_body_landed",
    "total_sig_strikes_body_attempted": "fighter_sig_str_body_attempted",
    "total_sig_strikes_leg_landed": "fighter_sig_str_leg_landed",
    "total_sig_strikes_leg_attempted": "fighter_sig_str_leg_attempted",
    "total_sig_strikes_distance_landed": "fighter_sig_str_distance_landed",
    "total_sig_strikes_distance_attempted": "fighter_sig_str_distance_attempted",
    "total_sig_strikes_clinch_landed": "fighter_sig_str_clinch_landed",
    "total_sig_strikes_clinch_attempted": "fighter_sig_str_clinch_attempted",
    "total_sig_strikes_ground_landed": "fighter_sig_str_ground_landed",
    "total_sig_strikes_ground_attempted": "fighter_sig_str_ground_attempted",
    "total_total_strikes_landed": "fighter_total_str_landed",
    "total_total_strikes_attempted": "fighter_total_str_attempted",
    "total_td_landed": "fighter_td_landed",
    "total_td_attempted": "fighter_td_attempted",
    "total_kd": "fighter_kd",
    "total_sub_att": "fighter_sub_att",
    "total_rev": "fighter_rev",
    "total_ctrl_seconds": "fighter_ctrl_seconds",
    "total_fight_duration_seconds": "fight_duration_seconds",
}

OUTCOME_CUMULATIVE_COLUMNS = [
    "total_ko_wins",
    "total_submission_wins",
    "total_decision_wins",
    "total_ko_losses",
    "total_losses",
]

ABSORBED_CUMULATIVE_COLUMNS = {
    "total_kd_absorbed": {"red": "blue_fighter_kd", "blue": "red_fighter_kd"},
    "total_td_absorbed": {"red": "blue_fighter_td_landed", "blue": "red_fighter_td_landed"},
    "total_td_attempted_against": {"red": "blue_fighter_td_attempted", "blue": "red_fighter_td_attempted"},
    "total_sig_strikes_head_absorbed": {
        "red": "blue_fighter_sig_str_head_landed",
        "blue": "red_fighter_sig_str_head_landed",
    },
    "total_sig_strikes_body_absorbed": {
        "red": "blue_fighter_sig_str_body_landed",
        "blue": "red_fighter_sig_str_body_landed",
    },
    "total_sig_strikes_leg_absorbed": {
        "red": "blue_fighter_sig_str_leg_landed",
        "blue": "red_fighter_sig_str_leg_landed",
    },
}

DERIVED_PREFIGHT_ACCURACY_COLUMNS = [
    "sig_strike_accuracy",
    "td_accuracy",
]

EPSILON = 1e-6
DEFAULT_ELO_INITIAL_RATING = 500.0
DEFAULT_ELO_K_FACTOR = 32.0
DEFAULT_LAST_N_RECENCY = 2
DEFAULT_EXP_DECAY_HALF_LIFE_DAYS = 365.0
WEIGHT_CLASS_LABELS = [
    "women's strawweight",
    "women's flyweight",
    "women's bantamweight",
    "women's featherweight",
    "light heavyweight",
    "middleweight",
    "welterweight",
    "lightweight",
    "featherweight",
    "bantamweight",
    "flyweight",
    "heavyweight",
    "open weight",
    "catch weight",
    "superfight",
]


def standardize_column_name(column: object) -> str:
    """Convert a column name to lowercase snake_case."""
    name = str(column).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized, unique column names."""
    renamed = df.copy()
    counts: dict[str, int] = {}
    columns: list[str] = []

    for column in renamed.columns:
        base = standardize_column_name(column) or "column"
        counts[base] = counts.get(base, 0) + 1
        columns.append(base if counts[base] == 1 else f"{base}_{counts[base]}")

    renamed.columns = columns
    return renamed


def detect_delimiter(path: Path) -> str:
    """Detect whether a CSV-like file is comma or semicolon separated."""
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        return ";" if first_line.count(";") > first_line.count(",") else ","


def load_csv_with_detected_delimiter(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV file after detecting comma vs semicolon delimiters."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.stat().st_size == 0:
        print(f"{label} file is empty: {path}")
        return pd.DataFrame()

    delimiter = detect_delimiter(path)
    df = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        na_values=list(PLACEHOLDERS),
        keep_default_na=True,
    )
    print(f"{label} shape: {df.shape}")
    print(f"{label} first columns: {df.columns[:8].tolist()}")
    print(f"{label} delimiter detected: {repr(delimiter)}")
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in object columns and normalize placeholders to NaN."""
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned.replace(list(PLACEHOLDERS), np.nan, inplace=True)
    return cleaned


def normalize_fighter_name(value: object) -> object:
    """Build a deterministic merge key for fighter names.

    This intentionally avoids fuzzy matching. It only removes casing and
    whitespace differences that should not represent different fighters.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized if normalized else pd.NA


def clean_weight_class(value: object) -> object:
    """Extract a compact weight-class label from bout type text."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    for label in WEIGHT_CLASS_LABELS:
        if label in text:
            if label == "superfight":
                return "Superfight"
            cleaned = label.title().replace("Women'S", "Women's")
            return cleaned

    return pd.NA


def parse_event_dates(series: pd.Series) -> pd.Series:
    """Parse event dates, preferring day-first dates used by the source file."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna() & series.notna()
    if not missing.any():
        return parsed

    fallback = pd.to_datetime(series[missing], errors="coerce")
    parsed.loc[missing] = fallback
    return parsed


def clean_fight_data(fights: pd.DataFrame) -> pd.DataFrame:
    """Clean fight-level rows and create the modeling target."""
    df = standardize_columns(fights)
    df = clean_text_columns(df)

    if "event_date" in df.columns:
        df["event_date"] = parse_event_dates(df["event_date"])
    else:
        df["event_date"] = pd.NaT

    if "fight_outcome" in df.columns:
        df["fight_outcome"] = df["fight_outcome"].astype("string").str.lower().str.strip()
        df["red_win"] = (df["fight_outcome"] == "red_win").astype("int64")
    else:
        df["red_win"] = 0

    required = ["red_fighter_name", "blue_fighter_name", "event_date"]
    existing_required = [column for column in required if column in df.columns]
    df = df.dropna(subset=existing_required).copy()

    for column in ["red_fighter_name", "blue_fighter_name"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    if "red_fighter_name" in df.columns:
        df["red_fighter_name_key"] = df["red_fighter_name"].apply(normalize_fighter_name)
    if "blue_fighter_name" in df.columns:
        df["blue_fighter_name_key"] = df["blue_fighter_name"].apply(normalize_fighter_name)
    if "bout_type" in df.columns:
        df["weight_class_clean"] = df["bout_type"].apply(clean_weight_class)

    return df


def parse_landed_attempted(value: object) -> tuple[float, float]:
    """Parse strings like '3 of 12' into landed and attempted counts."""
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip().lower()
    match = re.match(r"^(\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)$", text)
    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def parse_stat_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Split all known 'landed of attempted' columns into numeric features."""
    parsed = df.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        for suffix in STAT_PAIR_SUFFIXES:
            source = f"{corner}_fighter_{suffix}"
            if source not in parsed.columns:
                continue

            landed_col = f"{source}_landed"
            attempted_col = f"{source}_attempted"
            values = parsed[source].apply(parse_landed_attempted)

            parsed[landed_col] = values.apply(lambda item: item[0])
            parsed[attempted_col] = values.apply(lambda item: item[1])
            created_columns.extend([landed_col, attempted_col])

    return parsed, created_columns


def time_to_seconds(value: object) -> float:
    """Convert m:ss, h:mm:ss, or numeric time-like values into seconds."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text or text in PLACEHOLDERS:
        return np.nan

    if re.match(r"^\d+(?:\.\d+)?$", text):
        return float(text)

    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return np.nan

    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return float(minutes * 60 + seconds)
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return float(hours * 3600 + minutes * 60 + seconds)

    return np.nan


def convert_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create seconds-based columns for fight time and control time."""
    converted = df.copy()
    created_columns: list[str] = []

    if "time" in converted.columns:
        converted["fight_time_seconds"] = converted["time"].apply(time_to_seconds)
        created_columns.append("fight_time_seconds")

    for corner in ["red", "blue"]:
        source = f"{corner}_fighter_ctrl"
        if source in converted.columns:
            target = f"{corner}_fighter_ctrl_seconds"
            converted[target] = converted[source].apply(time_to_seconds)
            created_columns.append(target)

    if {"round", "fight_time_seconds"}.issubset(converted.columns):
        round_number = pd.to_numeric(converted["round"], errors="coerce")
        converted["total_fight_duration_seconds"] = ((round_number - 1) * 300) + converted[
            "fight_time_seconds"
        ]
        converted.loc[round_number.isna(), "total_fight_duration_seconds"] = np.nan
        created_columns.append("total_fight_duration_seconds")

    return converted, created_columns


def first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Return the first candidate column present in a dataframe."""
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def inches_from_height(value: object) -> float:
    """Convert heights like 5' 11\" to total inches when possible."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"')
    feet_match = re.search(r"(\d+)\s*(?:'|ft|feet)\s*(\d+)?", text)
    if feet_match:
        feet = int(feet_match.group(1))
        inches = int(feet_match.group(2) or 0)
        return float(feet * 12 + inches)

    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", text)
    if cm_match:
        return float(cm_match.group(1)) / 2.54

    numeric = re.search(r"\d+(?:\.\d+)?", text)
    return float(numeric.group(0)) if numeric else np.nan


def numeric_measurement(value: object) -> float:
    """Extract a numeric value from weight/reach strings."""
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def percentage_to_decimal(value: object) -> float:
    """Convert percentage strings like '55%' into decimals like 0.55."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text or text in PLACEHOLDERS:
        return np.nan

    if text.endswith("%"):
        numeric = numeric_measurement(text)
        return numeric / 100.0 if not pd.isna(numeric) else np.nan

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return np.nan

    numeric = float(numeric)
    return numeric / 100.0 if numeric > 1 else numeric


def clean_height(value: object, source_column: str | None) -> float:
    """Convert height values to inches, respecting cm-based source columns."""
    height = inches_from_height(value)
    if pd.isna(height):
        return np.nan
    if source_column and "cm" in source_column:
        return height / 2.54
    return height


def clean_weight(value: object, source_column: str | None) -> float:
    """Convert weight values to pounds, respecting kg-based source columns."""
    weight = numeric_measurement(value)
    if pd.isna(weight):
        return np.nan
    if source_column and "kg" in source_column:
        return weight * 2.2046226218
    return weight


def clean_reach(value: object, source_column: str | None) -> float:
    """Convert reach values to inches, respecting cm-based source columns."""
    reach = numeric_measurement(value)
    if pd.isna(reach):
        return np.nan
    if source_column and "cm" in source_column:
        return reach / 2.54
    return reach


def age_at_event_years(event_dates: pd.Series, dobs: pd.Series) -> pd.Series:
    """Calculate fighter age in full years at the fight date."""
    event_dates = pd.to_datetime(event_dates, errors="coerce")
    dobs = pd.to_datetime(dobs, errors="coerce")

    age = event_dates.dt.year - dobs.dt.year
    birthday_not_reached = (event_dates.dt.month < dobs.dt.month) | (
        (event_dates.dt.month == dobs.dt.month) & (event_dates.dt.day < dobs.dt.day)
    )
    age = age - birthday_not_reached.astype("Int64")
    age = age.where(event_dates.notna() & dobs.notna())
    return age.astype("Int64")


def clean_fighter_statistics(statistics: pd.DataFrame | None) -> pd.DataFrame:
    """Keep fighter profile attributes and source-provided rate stats."""
    if statistics is None or statistics.empty:
        return pd.DataFrame(
            columns=[
                "fighter_name",
                "fighter_name_key",
                "height",
                "weight",
                "reach",
                "stance",
                "dob",
                *FIGHTER_PROFILE_RATE_COLUMNS.keys(),
            ]
        )

    standardized = clean_text_columns(standardize_columns(statistics))
    selected = pd.DataFrame(index=standardized.index)
    source_columns: dict[str, str | None] = {}

    for target, candidates in STATIC_DETAIL_COLUMNS.items():
        source = first_existing_column(standardized.columns, candidates)
        source_columns[target] = source
        selected[target] = standardized[source] if source else np.nan

    for target, candidates in FIGHTER_PROFILE_RATE_COLUMNS.items():
        source = first_existing_column(standardized.columns, candidates)
        selected[target] = standardized[source] if source else np.nan

    selected = selected.dropna(subset=["fighter_name"]).copy()
    selected["fighter_name"] = selected["fighter_name"].astype("string").str.strip()
    selected["fighter_name_key"] = selected["fighter_name"].apply(normalize_fighter_name)
    selected["dob"] = parse_event_dates(selected["dob"])
    selected["height"] = selected["height"].apply(lambda value: clean_height(value, source_columns["height"]))
    selected["weight"] = selected["weight"].apply(lambda value: clean_weight(value, source_columns["weight"]))
    selected["reach"] = selected["reach"].apply(lambda value: clean_reach(value, source_columns["reach"]))
    selected["slpm"] = pd.to_numeric(selected["slpm"], errors="coerce")
    selected["sapm"] = pd.to_numeric(selected["sapm"], errors="coerce")
    selected["td_avg"] = pd.to_numeric(selected["td_avg"], errors="coerce")
    selected["sub_avg"] = pd.to_numeric(selected["sub_avg"], errors="coerce")
    selected["str_acc"] = selected["str_acc"].apply(percentage_to_decimal)
    selected["str_def"] = selected["str_def"].apply(percentage_to_decimal)
    selected["td_acc"] = selected["td_acc"].apply(percentage_to_decimal)
    selected["td_def"] = selected["td_def"].apply(percentage_to_decimal)
    selected = selected.dropna(subset=["fighter_name_key"]).copy()

    # If duplicate normalized names exist, keep the row with the most populated
    # static attributes. This stays deterministic without fuzzy matching.
    selected["_static_fields_present"] = selected[
        ["height", "weight", "reach", "stance", "dob", *FIGHTER_PROFILE_RATE_COLUMNS.keys()]
    ].notna().sum(axis=1)
    selected = selected.sort_values("_static_fields_present", ascending=False)
    selected = selected.drop_duplicates(subset=["fighter_name_key"], keep="first")
    selected = selected.drop(columns=["_static_fields_present"])

    return selected[
        ["fighter_name", "fighter_name_key", "height", "weight", "reach", "stance", "dob", *FIGHTER_PROFILE_RATE_COLUMNS.keys()]
    ]


# Backward-compatible name for earlier notebook imports.
clean_fighter_details = clean_fighter_statistics


def ensure_attribute_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ensure expected fighter attribute columns exist."""
    updated = df.copy()
    created_columns: list[str] = []

    for column in FIGHTER_ATTRIBUTE_COLUMNS:
        if column not in updated.columns:
            updated[column] = np.nan
            created_columns.append(column)

    return updated, created_columns


def get_fighter_attribute_missingness(df: pd.DataFrame) -> pd.Series:
    """Return missing-value percentages for key merged fighter attributes."""
    columns = [column for column in ATTRIBUTE_DIAGNOSTIC_COLUMNS if column in df.columns]
    if not columns:
        return pd.Series(dtype="float64")

    return df[columns].isna().mean().mul(100).round(2)


def get_unmatched_fighter_examples(
    fights: pd.DataFrame,
    fighter_statistics: pd.DataFrame,
    corner: str,
    limit: int = 10,
) -> list[str]:
    """Return example fighter names that do not have a details merge match."""
    key_column = f"{corner}_fighter_name_key"
    name_column = f"{corner}_fighter_name"
    if (
        fighter_statistics.empty
        or key_column not in fights.columns
        or "fighter_name_key" not in fighter_statistics.columns
    ):
        return []

    detail_keys = set(fighter_statistics["fighter_name_key"].dropna())
    unmatched = fights.loc[~fights[key_column].isin(detail_keys), name_column]
    return unmatched.dropna().drop_duplicates().head(limit).tolist()


def print_fighter_attribute_diagnostics(fights: pd.DataFrame, fighter_statistics: pd.DataFrame) -> None:
    """Print merge missingness and a few unmatched names for each corner."""
    missingness = get_fighter_attribute_missingness(fights)
    print("Merged fighter attribute missingness (%):")
    if missingness.empty:
        print("No fighter attribute columns are available for diagnostics.")
    else:
        print(missingness)

    for corner in ["red", "blue"]:
        examples = get_unmatched_fighter_examples(fights, fighter_statistics, corner)
        label = f"{corner.capitalize()} unmatched fighter examples"
        print(f"{label}: {examples if examples else 'None found or fighter statistics unavailable'}")


def merge_fighter_attributes(
    fights: pd.DataFrame,
    fighter_statistics: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge safe static fighter attributes onto red and blue corners using clean name keys."""
    merged = fights.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        name_column = f"{corner}_fighter_name"
        key_column = f"{corner}_fighter_name_key"
        if key_column not in merged.columns and name_column in merged.columns:
            merged[key_column] = merged[name_column].apply(normalize_fighter_name)
            created_columns.append(key_column)

    merged, empty_attribute_columns = ensure_attribute_columns(merged)
    created_columns.extend(empty_attribute_columns)

    if fighter_statistics.empty:
        print_fighter_attribute_diagnostics(merged, fighter_statistics)
        return merged, created_columns

    detail_attributes = fighter_statistics[
        ["fighter_name_key", "height", "weight", "reach", "stance", "dob", *FIGHTER_PROFILE_RATE_COLUMNS.keys()]
    ].copy()

    for corner in ["red", "blue"]:
        key_column = f"{corner}_fighter_name_key"
        if key_column not in merged.columns:
            continue

        rename_map = {
            "fighter_name_key": key_column,
            "height": f"__{corner}_fighter_height_merge",
            "weight": f"__{corner}_fighter_weight_merge",
            "reach": f"__{corner}_fighter_reach_merge",
            "stance": f"__{corner}_fighter_stance_merge",
            "dob": f"__{corner}_fighter_dob_merge",
            "slpm": f"__{corner}_fighter_slpm_merge",
            "str_acc": f"__{corner}_fighter_str_acc_merge",
            "sapm": f"__{corner}_fighter_sapm_merge",
            "str_def": f"__{corner}_fighter_str_def_merge",
            "td_avg": f"__{corner}_fighter_td_avg_merge",
            "td_acc": f"__{corner}_fighter_td_acc_merge",
            "td_def": f"__{corner}_fighter_td_def_merge",
            "sub_avg": f"__{corner}_fighter_sub_avg_merge",
        }
        corner_details = detail_attributes.rename(columns=rename_map)
        merged = merged.merge(corner_details, on=key_column, how="left")

        for attribute in ["height", "weight", "reach", "stance", "dob", *FIGHTER_PROFILE_RATE_COLUMNS.keys()]:
            target = f"{corner}_fighter_{attribute}"
            source = f"__{corner}_fighter_{attribute}_merge"
            merged[target] = merged[target].combine_first(merged[source])
            merged = merged.drop(columns=[source])

        dob_column = f"{corner}_fighter_dob"
        age_column = f"{corner}_fighter_age"
        merged[dob_column] = pd.to_datetime(merged[dob_column], errors="coerce")
        merged[age_column] = age_at_event_years(merged["event_date"], merged[dob_column])

    print_fighter_attribute_diagnostics(merged, fighter_statistics)
    return merged, created_columns


def get_numeric_value(row: pd.Series, column: str) -> float:
    """Safely fetch a numeric value from a fight row."""
    if column not in row.index or pd.isna(row[column]):
        return 0.0
    value = pd.to_numeric(row[column], errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def fighter_snapshot(totals: defaultdict[str, float]) -> dict[str, float]:
    """Return the current cumulative state for one fighter."""
    snapshot = {
        "total_fights": totals["total_fights"],
        "total_wins": totals["total_wins"],
    }
    for output_name in OUTCOME_CUMULATIVE_COLUMNS:
        snapshot[output_name] = totals[output_name]
    for output_name in CUMULATIVE_SOURCE_COLUMNS:
        snapshot[output_name] = totals[output_name]
    for output_name in ABSORBED_CUMULATIVE_COLUMNS:
        snapshot[output_name] = totals[output_name]
    return snapshot


def method_win_flags(method: object, won: int) -> dict[str, int]:
    """Return outcome-category counters for the current fight result."""
    if not won:
        return {
            "total_ko_wins": 0,
            "total_submission_wins": 0,
            "total_decision_wins": 0,
        }

    method_text = "" if pd.isna(method) else str(method).lower()
    return {
        "total_ko_wins": int("ko" in method_text),
        "total_submission_wins": int("submission" in method_text),
        "total_decision_wins": int("decision" in method_text),
    }


def method_loss_flags(method: object, lost: int) -> dict[str, int]:
    """Return loss-method counters for the current fight result."""
    if not lost:
        return {"total_ko_losses": 0}

    method_text = "" if pd.isna(method) else str(method).lower()
    return {"total_ko_losses": int("ko" in method_text)}


def add_prefight_efficiency_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create accuracy features from already-leakage-safe cumulative totals."""
    enriched = df.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        sig_landed = f"pre_fight_{corner}_total_sig_strikes_landed"
        sig_attempted = f"pre_fight_{corner}_total_sig_strikes_attempted"
        sig_accuracy = f"pre_fight_{corner}_sig_strike_accuracy"
        if {sig_landed, sig_attempted}.issubset(enriched.columns):
            enriched[sig_accuracy] = np.where(
                enriched[sig_attempted] > 0,
                enriched[sig_landed] / enriched[sig_attempted],
                0.0,
            )
            created_columns.append(sig_accuracy)

        td_landed = f"pre_fight_{corner}_total_td_landed"
        td_attempted = f"pre_fight_{corner}_total_td_attempted"
        td_accuracy = f"pre_fight_{corner}_td_accuracy"
        if {td_landed, td_attempted}.issubset(enriched.columns):
            enriched[td_accuracy] = np.where(
                enriched[td_attempted] > 0,
                enriched[td_landed] / enriched[td_attempted],
                0.0,
            )
            created_columns.append(td_accuracy)

    return enriched, created_columns


def add_prefight_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create exact-name pre-fight per-fight, per-minute, and efficiency columns."""
    enriched = df.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        total_fights = f"pre_fight_{corner}_total_fights"
        total_duration = f"pre_fight_{corner}_total_fight_duration_seconds"
        total_minutes = safe_ratio(enriched[total_duration], pd.Series(60.0, index=enriched.index), default=np.nan)

        per_fight_specs = [
            ("win_pct", "total_wins"),
            ("sig_strikes_landed_per_fight", "total_sig_strikes_landed"),
            ("sig_strikes_attempted_per_fight", "total_sig_strikes_attempted"),
            ("td_landed_per_fight", "total_td_landed"),
            ("td_attempted_per_fight", "total_td_attempted"),
            ("sub_att_per_fight", "total_sub_att"),
            ("ctrl_seconds_per_fight", "total_ctrl_seconds"),
            ("kd_per_fight", "total_kd"),
            ("kd_absorbed_per_fight", "total_kd_absorbed"),
        ]
        per_min_specs = [
            ("sig_strikes_landed_per_min", "total_sig_strikes_landed"),
            ("td_landed_per_min", "total_td_landed"),
            ("sub_att_per_min", "total_sub_att"),
            ("ctrl_seconds_per_min", "total_ctrl_seconds"),
        ]

        for feature_name, source_suffix in per_fight_specs:
            source_column = f"pre_fight_{corner}_{source_suffix}"
            target_column = f"pre_fight_{corner}_{feature_name}"
            if {source_column, total_fights}.issubset(enriched.columns):
                enriched[target_column] = safe_ratio(enriched[source_column], enriched[total_fights])
                created_columns.append(target_column)

        for feature_name, source_suffix in per_min_specs:
            source_column = f"pre_fight_{corner}_{source_suffix}"
            target_column = f"pre_fight_{corner}_{feature_name}"
            if {source_column, total_duration}.issubset(enriched.columns):
                enriched[target_column] = safe_ratio(
                    pd.to_numeric(enriched[source_column], errors="coerce").fillna(0.0) * 60.0,
                    enriched[total_duration],
                )
                created_columns.append(target_column)

        td_absorbed = f"pre_fight_{corner}_total_td_absorbed"
        td_attempted_against = f"pre_fight_{corner}_total_td_attempted_against"
        td_defense = f"pre_fight_{corner}_td_defense"
        if {td_absorbed, td_attempted_against}.issubset(enriched.columns):
            enriched[td_defense] = 1.0 - safe_ratio(enriched[td_absorbed], enriched[td_attempted_against])
            enriched.loc[pd.to_numeric(enriched[td_attempted_against], errors="coerce").fillna(0.0) <= 0, td_defense] = 0.0
            created_columns.append(td_defense)

        sig_accuracy = f"pre_fight_{corner}_sig_strike_accuracy"
        td_accuracy = f"pre_fight_{corner}_td_accuracy"
        if sig_accuracy not in enriched.columns and {
            f"pre_fight_{corner}_total_sig_strikes_landed",
            f"pre_fight_{corner}_total_sig_strikes_attempted",
        }.issubset(enriched.columns):
            enriched[sig_accuracy] = safe_ratio(
                enriched[f"pre_fight_{corner}_total_sig_strikes_landed"],
                enriched[f"pre_fight_{corner}_total_sig_strikes_attempted"],
            )
            created_columns.append(sig_accuracy)
        if td_accuracy not in enriched.columns and {
            f"pre_fight_{corner}_total_td_landed",
            f"pre_fight_{corner}_total_td_attempted",
        }.issubset(enriched.columns):
            enriched[td_accuracy] = safe_ratio(
                enriched[f"pre_fight_{corner}_total_td_landed"],
                enriched[f"pre_fight_{corner}_total_td_attempted"],
            )
            created_columns.append(td_accuracy)

    return enriched, created_columns


def safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Safely divide two numeric series with a consistent default."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(default, index=numerator.index, dtype="float64")
    valid = denominator.gt(0) & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def add_rate_and_efficiency_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create current-fight and pre-fight rate/efficiency features for major stats."""
    enriched = df.copy()
    created_columns: list[str] = []

    fight_minutes = safe_ratio(enriched["total_fight_duration_seconds"], pd.Series(60.0, index=enriched.index), default=np.nan)

    current_rate_columns = [
        ("sig_strikes_landed_per_minute", "fighter_sig_str_landed"),
        ("sig_strikes_attempted_per_minute", "fighter_sig_str_attempted"),
        ("total_strikes_landed_per_minute", "fighter_total_str_landed"),
        ("total_strikes_attempted_per_minute", "fighter_total_str_attempted"),
        ("td_landed_per_minute", "fighter_td_landed"),
        ("td_attempted_per_minute", "fighter_td_attempted"),
        ("kd_per_minute", "fighter_kd"),
        ("sub_att_per_minute", "fighter_sub_att"),
        ("rev_per_minute", "fighter_rev"),
    ]
    current_accuracy_columns = [
        ("sig_strike_accuracy", "fighter_sig_str_landed", "fighter_sig_str_attempted"),
        ("total_strike_accuracy", "fighter_total_str_landed", "fighter_total_str_attempted"),
        ("td_accuracy", "fighter_td_landed", "fighter_td_attempted"),
        ("sig_strikes_head_accuracy", "fighter_sig_str_head_landed", "fighter_sig_str_head_attempted"),
        ("sig_strikes_body_accuracy", "fighter_sig_str_body_landed", "fighter_sig_str_body_attempted"),
        ("sig_strikes_leg_accuracy", "fighter_sig_str_leg_landed", "fighter_sig_str_leg_attempted"),
        ("sig_strikes_distance_accuracy", "fighter_sig_str_distance_landed", "fighter_sig_str_distance_attempted"),
        ("sig_strikes_clinch_accuracy", "fighter_sig_str_clinch_landed", "fighter_sig_str_clinch_attempted"),
        ("sig_strikes_ground_accuracy", "fighter_sig_str_ground_landed", "fighter_sig_str_ground_attempted"),
    ]

    for corner in ["red", "blue"]:
        for feature_name, source_suffix in current_rate_columns:
            source_column = f"{corner}_{source_suffix}"
            target_column = f"{corner}_fighter_{feature_name}"
            if source_column in enriched.columns:
                enriched[target_column] = safe_ratio(enriched[source_column], fight_minutes)
                created_columns.append(target_column)

        ctrl_seconds = f"{corner}_fighter_ctrl_seconds"
        if ctrl_seconds in enriched.columns:
            share_column = f"{corner}_fighter_ctrl_share"
            per_minute_column = f"{corner}_fighter_ctrl_seconds_per_minute"
            enriched[share_column] = safe_ratio(enriched[ctrl_seconds], enriched["total_fight_duration_seconds"])
            enriched[per_minute_column] = safe_ratio(enriched[ctrl_seconds], fight_minutes)
            created_columns.extend([share_column, per_minute_column])

        for feature_name, landed_suffix, attempted_suffix in current_accuracy_columns:
            landed_column = f"{corner}_{landed_suffix}"
            attempted_column = f"{corner}_{attempted_suffix}"
            target_column = f"{corner}_fighter_{feature_name}"
            if {landed_column, attempted_column}.issubset(enriched.columns):
                enriched[target_column] = safe_ratio(enriched[landed_column], enriched[attempted_column])
                created_columns.append(target_column)

        prefight_duration = f"pre_fight_{corner}_total_fight_duration_seconds"
        if prefight_duration in enriched.columns:
            prefight_minutes = safe_ratio(enriched[prefight_duration], pd.Series(60.0, index=enriched.index), default=np.nan)
            prefight_rate_columns = [
                ("sig_strikes_landed_per_minute", "total_sig_strikes_landed"),
                ("sig_strikes_attempted_per_minute", "total_sig_strikes_attempted"),
                ("total_strikes_landed_per_minute", "total_total_strikes_landed"),
                ("total_strikes_attempted_per_minute", "total_total_strikes_attempted"),
                ("td_landed_per_minute", "total_td_landed"),
                ("td_attempted_per_minute", "total_td_attempted"),
                ("kd_per_minute", "total_kd"),
                ("sub_att_per_minute", "total_sub_att"),
                ("rev_per_minute", "total_rev"),
            ]
            for feature_name, source_suffix in prefight_rate_columns:
                source_column = f"pre_fight_{corner}_{source_suffix}"
                target_column = f"pre_fight_{corner}_{feature_name}"
                if source_column in enriched.columns:
                    enriched[target_column] = safe_ratio(enriched[source_column], prefight_minutes)
                    created_columns.append(target_column)

            ctrl_seconds = f"pre_fight_{corner}_total_ctrl_seconds"
            if ctrl_seconds in enriched.columns:
                share_column = f"pre_fight_{corner}_ctrl_share"
                per_minute_column = f"pre_fight_{corner}_ctrl_seconds_per_minute"
                enriched[share_column] = safe_ratio(enriched[ctrl_seconds], enriched[prefight_duration])
                enriched[per_minute_column] = safe_ratio(enriched[ctrl_seconds], prefight_minutes)
                created_columns.extend([share_column, per_minute_column])

            win_rate_column = f"pre_fight_{corner}_win_rate"
            enriched[win_rate_column] = safe_ratio(
                enriched[f"pre_fight_{corner}_total_wins"],
                enriched[f"pre_fight_{corner}_total_fights"],
            )
            created_columns.append(win_rate_column)

    return enriched, created_columns


def add_difference_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create red-minus-blue difference columns for paired numeric features."""
    enriched = df.copy()
    created_columns: list[str] = []
    difference_data: dict[str, pd.Series] = {}

    for column in list(enriched.columns):
        if not column.startswith("red_fighter_"):
            continue

        suffix = column.removeprefix("red_fighter_")
        blue_column = f"blue_fighter_{suffix}"
        if blue_column not in enriched.columns:
            continue
        if not (
            pd.api.types.is_numeric_dtype(enriched[column]) and pd.api.types.is_numeric_dtype(enriched[blue_column])
        ):
            continue

        diff_column = f"{suffix}_diff"
        difference_data[diff_column] = pd.to_numeric(enriched[column], errors="coerce") - pd.to_numeric(
            enriched[blue_column], errors="coerce"
        )
        created_columns.append(diff_column)

    for column in list(enriched.columns):
        if not column.startswith("pre_fight_red_"):
            continue

        suffix = column.removeprefix("pre_fight_red_")
        blue_column = f"pre_fight_blue_{suffix}"
        if blue_column not in enriched.columns:
            continue
        if not (
            pd.api.types.is_numeric_dtype(enriched[column]) and pd.api.types.is_numeric_dtype(enriched[blue_column])
        ):
            continue

        diff_column = f"pre_fight_{suffix}_diff"
        difference_data[diff_column] = pd.to_numeric(enriched[column], errors="coerce") - pd.to_numeric(
            enriched[blue_column], errors="coerce"
        )
        created_columns.append(diff_column)

    if difference_data:
        enriched = pd.concat([enriched, pd.DataFrame(difference_data, index=enriched.index)], axis=1)

    return enriched, created_columns


def stack_corner_values(df: pd.DataFrame, red_column: str, blue_column: str) -> pd.Series:
    """Stack red and blue fighter columns into a single numeric series."""
    values: list[pd.Series] = []
    if red_column in df.columns:
        values.append(pd.to_numeric(df[red_column], errors="coerce"))
    if blue_column in df.columns:
        values.append(pd.to_numeric(df[blue_column], errors="coerce"))
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True)


def compute_shared_zscores(
    df: pd.DataFrame,
    red_column: str,
    blue_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Compute red/blue z-scores using a shared stacked population."""
    stacked = stack_corner_values(df, red_column, blue_column)
    stacked = stacked.replace([np.inf, -np.inf], np.nan).dropna()
    if stacked.empty:
        return (
            pd.Series(0.0, index=df.index, dtype="float64"),
            pd.Series(0.0, index=df.index, dtype="float64"),
        )

    mean = stacked.mean()
    std = stacked.std(ddof=0)
    if pd.isna(std) or std < EPSILON:
        return (
            pd.Series(0.0, index=df.index, dtype="float64"),
            pd.Series(0.0, index=df.index, dtype="float64"),
        )

    red_values = pd.to_numeric(df[red_column], errors="coerce")
    blue_values = pd.to_numeric(df[blue_column], errors="coerce")
    return (red_values - mean) / std, (blue_values - mean) / std


def estimate_strike_type_weights(df: pd.DataFrame) -> dict[str, float]:
    """Estimate global strike-type weights from pre-fight historical patterns."""
    head = stack_corner_values(
        df,
        "pre_fight_red_total_sig_strikes_head_landed",
        "pre_fight_blue_total_sig_strikes_head_landed",
    )
    body = stack_corner_values(
        df,
        "pre_fight_red_total_sig_strikes_body_landed",
        "pre_fight_blue_total_sig_strikes_body_landed",
    )
    leg = stack_corner_values(
        df,
        "pre_fight_red_total_sig_strikes_leg_landed",
        "pre_fight_blue_total_sig_strikes_leg_landed",
    )
    kd = stack_corner_values(df, "pre_fight_red_total_kd", "pre_fight_blue_total_kd")

    valid = (
        head.notna()
        & body.notna()
        & leg.notna()
        & kd.notna()
        & ((head + body + leg) > 0)
    )
    if valid.sum() < 10:
        return {"head": 0.6, "body": 0.25, "leg": 0.15}

    head_valid = head[valid]
    body_valid = body[valid]
    leg_valid = leg[valid]
    kd_valid = kd[valid]

    # Blend correlation with historical strike share so the weights stay
    # data-driven without over-rewarding rare strike types.
    correlations = np.array(
        [
            max(head_valid.corr(kd_valid), 0.0),
            max(body_valid.corr(kd_valid), 0.0),
            max(leg_valid.corr(kd_valid), 0.0),
        ]
    )
    strike_shares = np.array(
        [
            head_valid.mean(),
            body_valid.mean(),
            leg_valid.mean(),
        ]
    )
    if strike_shares.sum() <= EPSILON:
        return {"head": 0.6, "body": 0.25, "leg": 0.15}

    strike_shares = strike_shares / strike_shares.sum()
    scores = correlations * strike_shares
    if scores.sum() <= EPSILON:
        return {"head": 0.6, "body": 0.25, "leg": 0.15}

    normalized = scores / scores.sum()
    return {"head": float(normalized[0]), "body": float(normalized[1]), "leg": float(normalized[2])}


def add_power_and_durability_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Create opponent-adjusted power and durability features from pre-fight history."""
    enriched = df.copy()
    created_columns: list[str] = []
    weights = estimate_strike_type_weights(enriched)

    for corner in ["red", "blue"]:
        head = f"pre_fight_{corner}_total_sig_strikes_head_landed"
        body = f"pre_fight_{corner}_total_sig_strikes_body_landed"
        leg = f"pre_fight_{corner}_total_sig_strikes_leg_landed"
        kd = f"pre_fight_{corner}_total_kd"
        total_fights = f"pre_fight_{corner}_total_fights"
        total_ko_wins = f"pre_fight_{corner}_total_ko_wins"
        total_ko_losses = f"pre_fight_{corner}_total_ko_losses"
        total_fight_duration = f"pre_fight_{corner}_total_fight_duration_seconds"

        weighted_strikes = f"pre_fight_{corner}_weighted_strikes"
        plain_eff = f"pre_fight_{corner}_kd_efficiency_plain"
        head_only_eff = f"pre_fight_{corner}_kd_efficiency_head_only"
        weighted_eff = f"pre_fight_{corner}_kd_efficiency_weighted"
        ko_rate = f"pre_fight_{corner}_ko_rate"
        kd_per_min = f"pre_fight_{corner}_kd_per_min"
        sig_strikes_per_min = f"pre_fight_{corner}_sig_strikes_per_min"

        total_plain_strikes = (
            pd.to_numeric(enriched[head], errors="coerce").fillna(0.0)
            + pd.to_numeric(enriched[body], errors="coerce").fillna(0.0)
            + pd.to_numeric(enriched[leg], errors="coerce").fillna(0.0)
        )
        enriched[weighted_strikes] = (
            weights["head"] * pd.to_numeric(enriched[head], errors="coerce").fillna(0.0)
            + weights["body"] * pd.to_numeric(enriched[body], errors="coerce").fillna(0.0)
            + weights["leg"] * pd.to_numeric(enriched[leg], errors="coerce").fillna(0.0)
        )
        enriched[plain_eff] = pd.to_numeric(enriched[kd], errors="coerce").fillna(0.0) / (total_plain_strikes + EPSILON)
        enriched[head_only_eff] = pd.to_numeric(enriched[kd], errors="coerce").fillna(0.0) / (
            pd.to_numeric(enriched[head], errors="coerce").fillna(0.0) + EPSILON
        )
        enriched[weighted_eff] = pd.to_numeric(enriched[kd], errors="coerce").fillna(0.0) / (
            enriched[weighted_strikes] + EPSILON
        )
        enriched[ko_rate] = safe_ratio(enriched[total_ko_wins], enriched[total_fights])
        enriched[kd_per_min] = safe_ratio(
            pd.to_numeric(enriched[kd], errors="coerce").fillna(0.0) * 60.0,
            enriched[total_fight_duration],
        )
        enriched[sig_strikes_per_min] = safe_ratio(
            pd.to_numeric(enriched[f"pre_fight_{corner}_total_sig_strikes_landed"], errors="coerce").fillna(0.0) * 60.0,
            enriched[total_fight_duration],
        )
        created_columns.extend(
            [weighted_strikes, plain_eff, head_only_eff, weighted_eff, ko_rate, kd_per_min, sig_strikes_per_min]
        )

        absorbed_head = f"pre_fight_{corner}_total_sig_strikes_head_absorbed"
        absorbed_body = f"pre_fight_{corner}_total_sig_strikes_body_absorbed"
        absorbed_leg = f"pre_fight_{corner}_total_sig_strikes_leg_absorbed"
        kd_absorbed = f"pre_fight_{corner}_total_kd_absorbed"
        weighted_absorbed = f"pre_fight_{corner}_weighted_strikes_absorbed"
        kd_absorbed_per_min = f"pre_fight_{corner}_kd_absorbed_per_min"
        kd_absorbed_eff = f"pre_fight_{corner}_kd_absorbed_efficiency"
        ko_loss_rate = f"pre_fight_{corner}_ko_loss_rate"

        enriched[weighted_absorbed] = (
            weights["head"] * pd.to_numeric(enriched[absorbed_head], errors="coerce").fillna(0.0)
            + weights["body"] * pd.to_numeric(enriched[absorbed_body], errors="coerce").fillna(0.0)
            + weights["leg"] * pd.to_numeric(enriched[absorbed_leg], errors="coerce").fillna(0.0)
        )
        enriched[kd_absorbed_per_min] = safe_ratio(
            pd.to_numeric(enriched[kd_absorbed], errors="coerce").fillna(0.0) * 60.0,
            enriched[total_fight_duration],
        )
        enriched[kd_absorbed_eff] = pd.to_numeric(enriched[kd_absorbed], errors="coerce").fillna(0.0) / (
            enriched[weighted_absorbed] + EPSILON
        )
        enriched[ko_loss_rate] = safe_ratio(enriched[total_ko_losses], enriched[total_fights])
        created_columns.extend([weighted_absorbed, kd_absorbed_per_min, kd_absorbed_eff, ko_loss_rate])

    power_z_specs = [
        ("pre_fight_red_kd_efficiency_weighted", "pre_fight_blue_kd_efficiency_weighted", "kd_efficiency_weighted"),
        ("pre_fight_red_kd_per_min", "pre_fight_blue_kd_per_min", "kd_per_min"),
        ("pre_fight_red_ko_rate", "pre_fight_blue_ko_rate", "ko_rate"),
    ]
    power_component_columns: dict[str, tuple[pd.Series, pd.Series]] = {}
    for red_col, blue_col, key in power_z_specs:
        power_component_columns[key] = compute_shared_zscores(enriched, red_col, blue_col)

    enriched["pre_fight_red_power_index"] = (
        power_component_columns["kd_efficiency_weighted"][0]
        + power_component_columns["kd_per_min"][0]
        + power_component_columns["ko_rate"][0]
    ) / 3.0
    enriched["pre_fight_blue_power_index"] = (
        power_component_columns["kd_efficiency_weighted"][1]
        + power_component_columns["kd_per_min"][1]
        + power_component_columns["ko_rate"][1]
    ) / 3.0
    created_columns.extend(["pre_fight_red_power_index", "pre_fight_blue_power_index"])

    vulnerability_z_specs = [
        ("pre_fight_red_kd_absorbed_efficiency", "pre_fight_blue_kd_absorbed_efficiency", "kd_absorbed_efficiency"),
        ("pre_fight_red_kd_absorbed_per_min", "pre_fight_blue_kd_absorbed_per_min", "kd_absorbed_per_min"),
        ("pre_fight_red_ko_loss_rate", "pre_fight_blue_ko_loss_rate", "ko_loss_rate"),
    ]
    vulnerability_component_columns: dict[str, tuple[pd.Series, pd.Series]] = {}
    for red_col, blue_col, key in vulnerability_z_specs:
        vulnerability_component_columns[key] = compute_shared_zscores(enriched, red_col, blue_col)

    enriched["pre_fight_red_vulnerability_index"] = (
        vulnerability_component_columns["kd_absorbed_efficiency"][0]
        + vulnerability_component_columns["kd_absorbed_per_min"][0]
        + vulnerability_component_columns["ko_loss_rate"][0]
    ) / 3.0
    enriched["pre_fight_blue_vulnerability_index"] = (
        vulnerability_component_columns["kd_absorbed_efficiency"][1]
        + vulnerability_component_columns["kd_absorbed_per_min"][1]
        + vulnerability_component_columns["ko_loss_rate"][1]
    ) / 3.0
    enriched["pre_fight_red_durability_index"] = -enriched["pre_fight_red_vulnerability_index"]
    enriched["pre_fight_blue_durability_index"] = -enriched["pre_fight_blue_vulnerability_index"]
    created_columns.extend(
        [
            "pre_fight_red_vulnerability_index",
            "pre_fight_blue_vulnerability_index",
            "pre_fight_red_durability_index",
            "pre_fight_blue_durability_index",
        ]
    )

    opponent_avg_columns = {
        "red": "pre_fight_red_avg_past_opponent_vulnerability",
        "blue": "pre_fight_blue_avg_past_opponent_vulnerability",
    }
    for column in opponent_avg_columns.values():
        enriched[column] = 0.0
        created_columns.append(column)

    opponent_histories: dict[str, list[float]] = defaultdict(list)
    working = enriched.copy()
    working["_opponent_adjustment_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_opponent_adjustment_order"], kind="mergesort")

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            for corner, fighter_name in [("red", row.get("red_fighter_name")), ("blue", row.get("blue_fighter_name"))]:
                history = opponent_histories[str(fighter_name)]
                working.at[index, opponent_avg_columns[corner]] = float(np.mean(history)) if history else 0.0

        for _, row in date_group.iterrows():
            red_name = str(row.get("red_fighter_name"))
            blue_name = str(row.get("blue_fighter_name"))
            red_vulnerability = pd.to_numeric(row.get("pre_fight_red_vulnerability_index"), errors="coerce")
            blue_vulnerability = pd.to_numeric(row.get("pre_fight_blue_vulnerability_index"), errors="coerce")
            if not pd.isna(blue_vulnerability):
                opponent_histories[red_name].append(float(blue_vulnerability))
            if not pd.isna(red_vulnerability):
                opponent_histories[blue_name].append(float(red_vulnerability))

    working = working.sort_values("_opponent_adjustment_order", kind="mergesort").drop(columns=["_opponent_adjustment_order"])
    enriched[opponent_avg_columns["red"]] = working[opponent_avg_columns["red"]]
    enriched[opponent_avg_columns["blue"]] = working[opponent_avg_columns["blue"]]

    enriched["pre_fight_red_opponent_adjusted_power"] = (
        enriched["pre_fight_red_power_index"] - enriched["pre_fight_red_avg_past_opponent_vulnerability"]
    )
    enriched["pre_fight_blue_opponent_adjusted_power"] = (
        enriched["pre_fight_blue_power_index"] - enriched["pre_fight_blue_avg_past_opponent_vulnerability"]
    )
    created_columns.extend(
        [
            "pre_fight_red_opponent_adjusted_power",
            "pre_fight_blue_opponent_adjusted_power",
        ]
    )

    explicit_diffs = {
        "diff_kd_efficiency_plain": (
            "pre_fight_red_kd_efficiency_plain",
            "pre_fight_blue_kd_efficiency_plain",
        ),
        "diff_kd_efficiency_head_only": (
            "pre_fight_red_kd_efficiency_head_only",
            "pre_fight_blue_kd_efficiency_head_only",
        ),
        "diff_kd_efficiency_weighted": (
            "pre_fight_red_kd_efficiency_weighted",
            "pre_fight_blue_kd_efficiency_weighted",
        ),
        "diff_ko_rate": ("pre_fight_red_ko_rate", "pre_fight_blue_ko_rate"),
        "diff_kd_per_min": ("pre_fight_red_kd_per_min", "pre_fight_blue_kd_per_min"),
        "diff_power_index": ("pre_fight_red_power_index", "pre_fight_blue_power_index"),
        "diff_durability_index": ("pre_fight_red_durability_index", "pre_fight_blue_durability_index"),
        "diff_opponent_adjusted_power": (
            "pre_fight_red_opponent_adjusted_power",
            "pre_fight_blue_opponent_adjusted_power",
        ),
        "diff_sig_strikes_per_min": (
            "pre_fight_red_sig_strikes_per_min",
            "pre_fight_blue_sig_strikes_per_min",
        ),
    }
    for diff_column, (red_column, blue_column) in explicit_diffs.items():
        if {red_column, blue_column}.issubset(enriched.columns):
            enriched[diff_column] = pd.to_numeric(enriched[red_column], errors="coerce") - pd.to_numeric(
                enriched[blue_column], errors="coerce"
            )
            created_columns.append(diff_column)

    return enriched, created_columns, weights


def add_prefight_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add streak, opponent-strength, and UFC-experience features without leakage."""
    working = df.copy()
    created_columns: list[str] = []
    working["_context_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_context_order"], kind="mergesort")

    streak_state: dict[str, dict[str, float]] = defaultdict(lambda: {"win_streak": 0.0, "loss_streak": 0.0})
    fighter_first_fight_date: dict[str, pd.Timestamp] = {}
    opponent_histories: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    required_output_columns = [
        "pre_fight_red_win_streak",
        "pre_fight_red_loss_streak",
        "pre_fight_blue_win_streak",
        "pre_fight_blue_loss_streak",
        "pre_fight_red_avg_past_opponent_win_pct",
        "pre_fight_blue_avg_past_opponent_win_pct",
        "pre_fight_red_avg_past_opponent_total_wins",
        "pre_fight_blue_avg_past_opponent_total_wins",
        "pre_fight_red_avg_past_opponent_total_fights",
        "pre_fight_blue_avg_past_opponent_total_fights",
        "pre_fight_red_years_in_ufc",
        "pre_fight_blue_years_in_ufc",
        "pre_fight_red_fights_per_year",
        "pre_fight_blue_fights_per_year",
    ]
    optional_output_columns = []
    if {"pre_fight_red_power_index", "pre_fight_blue_power_index"}.issubset(working.columns):
        optional_output_columns.extend(
            ["pre_fight_red_avg_past_opponent_power_index", "pre_fight_blue_avg_past_opponent_power_index"]
        )
    if {"pre_fight_red_durability_index", "pre_fight_blue_durability_index"}.issubset(working.columns):
        optional_output_columns.extend(
            [
                "pre_fight_red_avg_past_opponent_durability_index",
                "pre_fight_blue_avg_past_opponent_durability_index",
            ]
        )
    for column in [*required_output_columns, *optional_output_columns]:
        working[column] = 0.0
        created_columns.append(column)

    metric_specs = [
        ("avg_past_opponent_win_pct", "pre_fight_red_win_pct", "pre_fight_blue_win_pct"),
        ("avg_past_opponent_total_wins", "pre_fight_red_total_wins", "pre_fight_blue_total_wins"),
        ("avg_past_opponent_total_fights", "pre_fight_red_total_fights", "pre_fight_blue_total_fights"),
    ]
    if {"pre_fight_red_power_index", "pre_fight_blue_power_index"}.issubset(working.columns):
        metric_specs.append(("avg_past_opponent_power_index", "pre_fight_red_power_index", "pre_fight_blue_power_index"))
    if {"pre_fight_red_durability_index", "pre_fight_blue_durability_index"}.issubset(working.columns):
        metric_specs.append(
            ("avg_past_opponent_durability_index", "pre_fight_red_durability_index", "pre_fight_blue_durability_index")
        )

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
            for corner in ["red", "blue"]:
                fighter_name = row.get(f"{corner}_fighter_name")
                if pd.isna(fighter_name):
                    continue

                fighter_name = str(fighter_name)
                streaks = streak_state[fighter_name]
                working.at[index, f"pre_fight_{corner}_win_streak"] = streaks["win_streak"]
                working.at[index, f"pre_fight_{corner}_loss_streak"] = streaks["loss_streak"]

                first_fight_date = fighter_first_fight_date.get(fighter_name)
                days_since_first = 0.0
                if first_fight_date is not None and not pd.isna(event_date):
                    days_since_first = max((event_date - first_fight_date).days, 0)

                working.at[index, f"pre_fight_{corner}_years_in_ufc"] = days_since_first / 365.25 if days_since_first else 0.0
                total_fights = pd.to_numeric(row.get(f"pre_fight_{corner}_total_fights"), errors="coerce")
                total_fights = 0.0 if pd.isna(total_fights) else float(total_fights)
                fights_per_year = (total_fights * 365.25 / days_since_first) if days_since_first > 0 else 0.0
                working.at[index, f"pre_fight_{corner}_fights_per_year"] = fights_per_year

                for metric_name, _, _ in metric_specs:
                    history = opponent_histories[fighter_name][metric_name]
                    target = f"pre_fight_{corner}_{metric_name}"
                    working.at[index, target] = float(np.mean(history)) if history else 0.0

        for _, row in date_group.iterrows():
            red_name = row.get("red_fighter_name")
            blue_name = row.get("blue_fighter_name")
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
            if not pd.isna(red_name) and str(red_name) not in fighter_first_fight_date and not pd.isna(event_date):
                fighter_first_fight_date[str(red_name)] = event_date
            if not pd.isna(blue_name) and str(blue_name) not in fighter_first_fight_date and not pd.isna(event_date):
                fighter_first_fight_date[str(blue_name)] = event_date

            fight_outcome = str(row.get("fight_outcome", "")).lower()
            if not pd.isna(red_name):
                red_name = str(red_name)
            if not pd.isna(blue_name):
                blue_name = str(blue_name)

            if fight_outcome == "red_win":
                streak_state[red_name]["win_streak"] += 1
                streak_state[red_name]["loss_streak"] = 0.0
                streak_state[blue_name]["loss_streak"] += 1
                streak_state[blue_name]["win_streak"] = 0.0
            elif fight_outcome == "blue_win":
                streak_state[blue_name]["win_streak"] += 1
                streak_state[blue_name]["loss_streak"] = 0.0
                streak_state[red_name]["loss_streak"] += 1
                streak_state[red_name]["win_streak"] = 0.0

            for metric_name, red_source, blue_source in metric_specs:
                blue_value = pd.to_numeric(row.get(blue_source), errors="coerce")
                red_value = pd.to_numeric(row.get(red_source), errors="coerce")
                if not pd.isna(red_name) and not pd.isna(blue_value):
                    opponent_histories[red_name][metric_name].append(float(blue_value))
                if not pd.isna(blue_name) and not pd.isna(red_value):
                    opponent_histories[blue_name][metric_name].append(float(red_value))

    working = working.sort_values("_context_order", kind="mergesort").drop(columns=["_context_order"])

    explicit_diffs = {
        "diff_win_streak": ("pre_fight_red_win_streak", "pre_fight_blue_win_streak"),
        "diff_loss_streak": ("pre_fight_red_loss_streak", "pre_fight_blue_loss_streak"),
        "diff_avg_past_opponent_win_pct": (
            "pre_fight_red_avg_past_opponent_win_pct",
            "pre_fight_blue_avg_past_opponent_win_pct",
        ),
        "diff_avg_past_opponent_total_wins": (
            "pre_fight_red_avg_past_opponent_total_wins",
            "pre_fight_blue_avg_past_opponent_total_wins",
        ),
        "diff_avg_past_opponent_total_fights": (
            "pre_fight_red_avg_past_opponent_total_fights",
            "pre_fight_blue_avg_past_opponent_total_fights",
        ),
        "diff_years_in_ufc": ("pre_fight_red_years_in_ufc", "pre_fight_blue_years_in_ufc"),
        "diff_fights_per_year": ("pre_fight_red_fights_per_year", "pre_fight_blue_fights_per_year"),
    }
    for diff_column, (red_column, blue_column) in explicit_diffs.items():
        if {red_column, blue_column}.issubset(working.columns):
            working[diff_column] = pd.to_numeric(working[red_column], errors="coerce") - pd.to_numeric(
                working[blue_column], errors="coerce"
            )
            created_columns.append(diff_column)

    return working, created_columns


def add_explicit_difference_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create short diff_* aliases for common pre-fight and physical comparison columns."""
    enriched = df.copy()
    created_columns: list[str] = []
    alias_map = {
        "diff_total_fights": "pre_fight_total_fights_diff",
        "diff_total_wins": "pre_fight_total_wins_diff",
        "diff_total_losses": "pre_fight_total_losses_diff",
        "diff_sig_strikes_landed": "pre_fight_total_sig_strikes_landed_diff",
        "diff_sig_strikes_attempted": "pre_fight_total_sig_strikes_attempted_diff",
        "diff_total_strikes_landed": "pre_fight_total_total_strikes_landed_diff",
        "diff_total_strikes_attempted": "pre_fight_total_total_strikes_attempted_diff",
        "diff_td_landed": "pre_fight_total_td_landed_diff",
        "diff_td_attempted": "pre_fight_total_td_attempted_diff",
        "diff_total_kd": "pre_fight_total_kd_diff",
        "diff_sub_att": "pre_fight_total_sub_att_diff",
        "diff_ctrl_seconds": "pre_fight_total_ctrl_seconds_diff",
        "diff_win_pct": "pre_fight_win_pct_diff",
        "diff_sig_strikes_landed_per_fight": "pre_fight_sig_strikes_landed_per_fight_diff",
        "diff_sig_strikes_attempted_per_fight": "pre_fight_sig_strikes_attempted_per_fight_diff",
        "diff_sig_strikes_landed_per_min": "pre_fight_sig_strikes_landed_per_min_diff",
        "diff_td_landed_per_fight": "pre_fight_td_landed_per_fight_diff",
        "diff_td_attempted_per_fight": "pre_fight_td_attempted_per_fight_diff",
        "diff_td_landed_per_min": "pre_fight_td_landed_per_min_diff",
        "diff_sub_att_per_fight": "pre_fight_sub_att_per_fight_diff",
        "diff_sub_att_per_min": "pre_fight_sub_att_per_min_diff",
        "diff_ctrl_seconds_per_fight": "pre_fight_ctrl_seconds_per_fight_diff",
        "diff_ctrl_seconds_per_min": "pre_fight_ctrl_seconds_per_min_diff",
        "diff_kd_per_fight": "pre_fight_kd_per_fight_diff",
        "diff_kd_absorbed_per_fight": "pre_fight_kd_absorbed_per_fight_diff",
        "diff_td_accuracy": "pre_fight_td_accuracy_diff",
        "diff_sig_strike_accuracy": "pre_fight_sig_strike_accuracy_diff",
        "diff_td_defense": "pre_fight_td_defense_diff",
        "diff_age": "age_diff",
        "diff_reach": "reach_diff",
        "diff_height": "height_diff",
    }
    for alias, source in alias_map.items():
        if source in enriched.columns:
            enriched[alias] = pd.to_numeric(enriched[source], errors="coerce")
            created_columns.append(alias)

    return enriched, created_columns


def build_elo_features(
    df: pd.DataFrame,
    k_factor: float = DEFAULT_ELO_K_FACTOR,
    initial_rating: float = DEFAULT_ELO_INITIAL_RATING,
) -> tuple[pd.DataFrame, list[str]]:
    """Create strictly pre-fight Elo ratings and update them only after each fight.

    Every fighter starts at the same baseline Elo. That keeps the implementation
    clean and leakage-safe, but it does mean experienced newcomers from other
    promotions are treated like UFC first-timers in the first version.
    """
    working = df.copy()
    created_columns = [
        "pre_fight_red_elo",
        "pre_fight_blue_elo",
        "diff_elo",
        "pre_fight_red_elo_win_prob",
        "pre_fight_blue_elo_win_prob",
    ]
    working["_elo_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_elo_order"], kind="mergesort")

    ratings: defaultdict[str, float] = defaultdict(lambda: float(initial_rating))
    for column in created_columns:
        working[column] = np.nan

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            red_name = str(row.get("red_fighter_name"))
            blue_name = str(row.get("blue_fighter_name"))
            red_rating = float(ratings[red_name])
            blue_rating = float(ratings[blue_name])
            red_expected = 1.0 / (1.0 + 10.0 ** ((blue_rating - red_rating) / 400.0))
            blue_expected = 1.0 - red_expected

            working.at[index, "pre_fight_red_elo"] = red_rating
            working.at[index, "pre_fight_blue_elo"] = blue_rating
            working.at[index, "diff_elo"] = red_rating - blue_rating
            working.at[index, "pre_fight_red_elo_win_prob"] = red_expected
            working.at[index, "pre_fight_blue_elo_win_prob"] = blue_expected

        for _, row in date_group.iterrows():
            red_name = str(row.get("red_fighter_name"))
            blue_name = str(row.get("blue_fighter_name"))
            red_rating = float(ratings[red_name])
            blue_rating = float(ratings[blue_name])
            red_expected = 1.0 / (1.0 + 10.0 ** ((blue_rating - red_rating) / 400.0))
            blue_expected = 1.0 - red_expected

            fight_outcome = str(row.get("fight_outcome", "")).lower()
            if fight_outcome == "red_win":
                red_score, blue_score = 1.0, 0.0
            elif fight_outcome == "blue_win":
                red_score, blue_score = 0.0, 1.0
            else:
                red_score = blue_score = 0.5

            ratings[red_name] = red_rating + k_factor * (red_score - red_expected)
            ratings[blue_name] = blue_rating + k_factor * (blue_score - blue_expected)

    working = working.sort_values("_elo_order", kind="mergesort").drop(columns=["_elo_order"])
    return working, created_columns


def print_elo_diagnostics(
    df: pd.DataFrame,
    feature_columns: list[str],
    initial_rating: float = DEFAULT_ELO_INITIAL_RATING,
) -> None:
    """Print a compact set of Elo sanity checks and previews."""
    unique_columns = list(dict.fromkeys(feature_columns))
    preview_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        "pre_fight_red_elo",
        "pre_fight_blue_elo",
        "diff_elo",
        "pre_fight_red_elo_win_prob",
        "pre_fight_blue_elo_win_prob",
        "red_win",
    ]
    available_preview = [column for column in preview_columns if column in df.columns]

    newcomer_red_mask = df.get("pre_fight_red_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    newcomer_blue_mask = df.get("pre_fight_blue_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    red_newcomer_mismatches = int(
        (~np.isclose(pd.to_numeric(df.loc[newcomer_red_mask, "pre_fight_red_elo"], errors="coerce"), initial_rating)).sum()
    ) if "pre_fight_red_elo" in df.columns else 0
    blue_newcomer_mismatches = int(
        (~np.isclose(pd.to_numeric(df.loc[newcomer_blue_mask, "pre_fight_blue_elo"], errors="coerce"), initial_rating)).sum()
    ) if "pre_fight_blue_elo" in df.columns else 0
    diff_check = int(
        (~np.isclose(
            pd.to_numeric(df.get("diff_elo"), errors="coerce"),
            pd.to_numeric(df.get("pre_fight_red_elo"), errors="coerce")
            - pd.to_numeric(df.get("pre_fight_blue_elo"), errors="coerce"),
            equal_nan=True,
        )).sum()
    ) if {"diff_elo", "pre_fight_red_elo", "pre_fight_blue_elo"}.issubset(df.columns) else 0

    print("New Elo columns:")
    print(unique_columns)
    print("Elo sanity checks:")
    print(f"- red newcomer Elo mismatches from {initial_rating:.0f}: {red_newcomer_mismatches}")
    print(f"- blue newcomer Elo mismatches from {initial_rating:.0f}: {blue_newcomer_mismatches}")
    print(f"- diff_elo consistency mismatches: {diff_check}")
    print("Elo preview (first 5 rows):")
    print(df[available_preview].head())


def build_fight_history_entry(row: pd.Series, corner: str) -> dict[str, float | pd.Timestamp]:
    """Extract one fighter's current-fight stats for later recency summaries."""
    opponent_corner = "blue" if corner == "red" else "red"
    duration_seconds = get_numeric_value(row, "total_fight_duration_seconds")
    fight_outcome = str(row.get("fight_outcome", "")).lower()

    return {
        "event_date": pd.to_datetime(row.get("event_date"), errors="coerce"),
        "won": float(fight_outcome == f"{corner}_win"),
        "sig_strikes_landed": get_numeric_value(row, f"{corner}_fighter_sig_str_landed"),
        "td_landed": get_numeric_value(row, f"{corner}_fighter_td_landed"),
        "ctrl_seconds": get_numeric_value(row, f"{corner}_fighter_ctrl_seconds"),
        "kd": get_numeric_value(row, f"{corner}_fighter_kd"),
        "kd_absorbed": get_numeric_value(row, f"{opponent_corner}_fighter_kd"),
        "fight_duration_seconds": duration_seconds,
    }


def empty_recency_snapshot() -> dict[str, float]:
    """Return zero-filled recency metrics for fighters without prior UFC history."""
    return {
        "win_pct": 0.0,
        "sig_strikes_landed_per_fight": 0.0,
        "sig_strikes_landed_per_min": 0.0,
        "td_landed_per_fight": 0.0,
        "ctrl_seconds_per_fight": 0.0,
        "kd_per_fight": 0.0,
        "kd_absorbed_per_fight": 0.0,
    }


def summarize_weighted_history(entries: list[dict[str, float | pd.Timestamp]], weights: np.ndarray) -> dict[str, float]:
    """Summarize fight-history entries with arbitrary non-negative weights.

    When a fighter has no prior UFC fights we return zeros consistently so the
    downstream modeling pipeline can keep the first-fight rows without special
    casing or leakage-prone imputations.
    """
    if not entries:
        return empty_recency_snapshot()

    weights = np.asarray(weights, dtype="float64")
    if weights.size != len(entries) or np.nansum(weights) <= EPSILON:
        return empty_recency_snapshot()

    weight_sum = float(np.nansum(weights))
    weighted_duration_seconds = float(
        sum(weight * float(entry.get("fight_duration_seconds", 0.0) or 0.0) for weight, entry in zip(weights, entries))
    )

    def weighted_average(key: str) -> float:
        numerator = float(sum(weight * float(entry.get(key, 0.0) or 0.0) for weight, entry in zip(weights, entries)))
        return numerator / weight_sum if weight_sum > EPSILON else 0.0

    weighted_sig_landed = float(
        sum(weight * float(entry.get("sig_strikes_landed", 0.0) or 0.0) for weight, entry in zip(weights, entries))
    )

    snapshot = {
        "win_pct": weighted_average("won"),
        "sig_strikes_landed_per_fight": weighted_average("sig_strikes_landed"),
        "sig_strikes_landed_per_min": (weighted_sig_landed * 60.0 / weighted_duration_seconds) if weighted_duration_seconds > EPSILON else 0.0,
        "td_landed_per_fight": weighted_average("td_landed"),
        "ctrl_seconds_per_fight": weighted_average("ctrl_seconds"),
        "kd_per_fight": weighted_average("kd"),
        "kd_absorbed_per_fight": weighted_average("kd_absorbed"),
    }
    return snapshot


def build_last_n_recency_features(
    df: pd.DataFrame,
    n_last: int = DEFAULT_LAST_N_RECENCY,
) -> tuple[pd.DataFrame, list[str]]:
    """Create strictly pre-fight last-N recency features from prior UFC fights only."""
    if n_last <= 0:
        raise ValueError("n_last must be a positive integer")

    working = df.copy()
    working["_recent_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_recent_order"], kind="mergesort")

    metric_suffixes = list(empty_recency_snapshot().keys())
    created_columns: list[str] = []
    for corner in ["red", "blue"]:
        for metric_suffix in metric_suffixes:
            column = f"recent_{n_last}_{corner}_{metric_suffix}"
            working[column] = 0.0
            created_columns.append(column)
    for metric_suffix in metric_suffixes:
        diff_column = f"diff_recent_{n_last}_{metric_suffix}"
        working[diff_column] = 0.0
        created_columns.append(diff_column)

    fighter_histories: defaultdict[str, list[dict[str, float | pd.Timestamp]]] = defaultdict(list)

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            for corner in ["red", "blue"]:
                fighter_name = str(row.get(f"{corner}_fighter_name"))
                history_window = fighter_histories[fighter_name][-n_last:]
                snapshot = summarize_weighted_history(history_window, np.ones(len(history_window), dtype="float64"))
                for metric_suffix, value in snapshot.items():
                    working.at[index, f"recent_{n_last}_{corner}_{metric_suffix}"] = value

        for metric_suffix in metric_suffixes:
            red_column = f"recent_{n_last}_red_{metric_suffix}"
            blue_column = f"recent_{n_last}_blue_{metric_suffix}"
            diff_column = f"diff_recent_{n_last}_{metric_suffix}"
            date_indices = date_group.index
            working.loc[date_indices, diff_column] = (
                pd.to_numeric(working.loc[date_indices, red_column], errors="coerce")
                - pd.to_numeric(working.loc[date_indices, blue_column], errors="coerce")
            )

        for _, row in date_group.iterrows():
            for corner in ["red", "blue"]:
                fighter_name = str(row.get(f"{corner}_fighter_name"))
                fighter_histories[fighter_name].append(build_fight_history_entry(row, corner))

    working = working.sort_values("_recent_order", kind="mergesort").drop(columns=["_recent_order"])
    return working, created_columns


def build_exp_decay_recency_features(
    df: pd.DataFrame,
    half_life_days: float = DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Create pre-fight exponentially decayed recency features from prior UFC fights."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    working = df.copy()
    working["_exp_decay_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_exp_decay_order"], kind="mergesort")

    metric_suffixes = list(empty_recency_snapshot().keys())
    created_columns: list[str] = []
    for corner in ["red", "blue"]:
        for metric_suffix in metric_suffixes:
            column = f"exp_decay_{corner}_{metric_suffix}"
            working[column] = 0.0
            created_columns.append(column)
    for metric_suffix in metric_suffixes:
        diff_column = f"diff_exp_decay_{metric_suffix}"
        working[diff_column] = 0.0
        created_columns.append(diff_column)

    fighter_histories: defaultdict[str, list[dict[str, float | pd.Timestamp]]] = defaultdict(list)
    decay_lambda = float(np.log(2.0) / half_life_days)

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            current_date = pd.to_datetime(row.get("event_date"), errors="coerce")
            for corner in ["red", "blue"]:
                fighter_name = str(row.get(f"{corner}_fighter_name"))
                history = fighter_histories[fighter_name]
                if not history or pd.isna(current_date):
                    snapshot = empty_recency_snapshot()
                else:
                    weights: list[float] = []
                    for entry in history:
                        past_date = pd.to_datetime(entry.get("event_date"), errors="coerce")
                        days_since = max(float((current_date - past_date).days), 0.0) if not pd.isna(past_date) else 0.0
                        weights.append(float(np.exp(-decay_lambda * days_since)))
                    snapshot = summarize_weighted_history(history, np.asarray(weights, dtype="float64"))

                for metric_suffix, value in snapshot.items():
                    working.at[index, f"exp_decay_{corner}_{metric_suffix}"] = value

        for metric_suffix in metric_suffixes:
            red_column = f"exp_decay_red_{metric_suffix}"
            blue_column = f"exp_decay_blue_{metric_suffix}"
            diff_column = f"diff_exp_decay_{metric_suffix}"
            date_indices = date_group.index
            working.loc[date_indices, diff_column] = (
                pd.to_numeric(working.loc[date_indices, red_column], errors="coerce")
                - pd.to_numeric(working.loc[date_indices, blue_column], errors="coerce")
            )

        for _, row in date_group.iterrows():
            for corner in ["red", "blue"]:
                fighter_name = str(row.get(f"{corner}_fighter_name"))
                fighter_histories[fighter_name].append(build_fight_history_entry(row, corner))

    working = working.sort_values("_exp_decay_order", kind="mergesort").drop(columns=["_exp_decay_order"])
    return working, created_columns


def print_recency_feature_diagnostics(
    df: pd.DataFrame,
    last_n_columns: list[str],
    exp_decay_columns: list[str],
    n_last: int,
    half_life_days: float,
) -> None:
    """Print leakage-safety checks and a compact preview for recency features."""
    preview_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        f"recent_{n_last}_red_win_pct",
        f"recent_{n_last}_blue_win_pct",
        f"diff_recent_{n_last}_win_pct",
        "exp_decay_red_win_pct",
        "exp_decay_blue_win_pct",
        "diff_exp_decay_win_pct",
        "diff_elo",
    ]
    available_preview = [column for column in preview_columns if column in df.columns]

    red_first = df.get("pre_fight_red_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    blue_first = df.get("pre_fight_blue_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    red_recent_mismatch = int((pd.to_numeric(df.loc[red_first, f"recent_{n_last}_red_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if f"recent_{n_last}_red_win_pct" in df.columns else 0
    blue_recent_mismatch = int((pd.to_numeric(df.loc[blue_first, f"recent_{n_last}_blue_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if f"recent_{n_last}_blue_win_pct" in df.columns else 0
    red_exp_mismatch = int((pd.to_numeric(df.loc[red_first, "exp_decay_red_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if "exp_decay_red_win_pct" in df.columns else 0
    blue_exp_mismatch = int((pd.to_numeric(df.loc[blue_first, "exp_decay_blue_win_pct"], errors="coerce").fillna(0.0) != 0.0).sum()) if "exp_decay_blue_win_pct" in df.columns else 0

    sample_days = np.array([30.0, 180.0, 365.0, 730.0], dtype="float64")
    sample_weights = np.exp(-(np.log(2.0) / half_life_days) * sample_days)
    monotonic_decay = bool(np.all(np.diff(sample_weights) < 0))

    print("New last-N recency columns:")
    print(list(dict.fromkeys(last_n_columns)))
    print("New exponential-decay recency columns:")
    print(list(dict.fromkeys(exp_decay_columns)))
    print("Recency sanity checks:")
    print(f"- red first-fight last-{n_last} history mismatches: {red_recent_mismatch}")
    print(f"- blue first-fight last-{n_last} history mismatches: {blue_recent_mismatch}")
    print(f"- red first-fight exp-decay history mismatches: {red_exp_mismatch}")
    print(f"- blue first-fight exp-decay history mismatches: {blue_exp_mismatch}")
    print(f"- exp-decay weights decrease with age: {monotonic_decay}")
    print(f"- sample exp-decay weights for {sample_days.astype(int).tolist()} days: {sample_weights.round(4).tolist()}")
    print("Recency preview (first 5 rows):")
    print(df[available_preview].head())


def coalesce_numeric_columns(df: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    """Return the first available numeric value across candidate columns."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        result = result.combine_first(candidate)
    return result.fillna(default)


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    """Apply log1p after clipping negatives that should not contribute to volume-style scores."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.log1p(numeric.clip(lower=0.0))


def parse_scheduled_rounds(value: object) -> float:
    """Extract the scheduled number of rounds from a time-format style string."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan

    leading_match = re.match(r"^\s*(\d+)\s*rnd", text)
    if leading_match:
        return float(leading_match.group(1))

    parenthetical = re.search(r"\(([\d\-\s]+)\)", text)
    if parenthetical:
        values = [part for part in parenthetical.group(1).split("-") if part.strip()]
        if values:
            return float(len(values))

    digits = re.search(r"\d+", text)
    return float(digits.group(0)) if digits else np.nan


def add_style_and_matchup_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create human-hypothesis style and matchup interaction features from pre-fight stats."""
    enriched = df.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        opponent_corner = "blue" if corner == "red" else "red"
        strike_volume_source = coalesce_numeric_columns(
            enriched,
            [
                f"pre_fight_{corner}_sig_strikes_landed_per_min",
                f"pre_fight_{corner}_sig_strikes_landed_per_minute",
            ],
        )
        strike_per_fight_source = coalesce_numeric_columns(
            enriched,
            [f"pre_fight_{corner}_sig_strikes_landed_per_fight"],
        )
        kd_per_fight_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_kd_per_fight"])
        kd_per_min_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_kd_per_min"])
        ko_rate_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_ko_rate"])
        power_index_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_power_index"])
        td_attempted_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_td_attempted_per_fight"])
        td_landed_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_td_landed_per_fight"])
        ctrl_per_fight_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_ctrl_seconds_per_fight"])
        sub_per_fight_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_sub_att_per_fight"])
        td_accuracy_source = coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_td_accuracy"])
        opponent_td_defense_source = coalesce_numeric_columns(enriched, [f"pre_fight_{opponent_corner}_td_defense"])

        striking_volume_score = (
            log1p_nonnegative(strike_volume_source) + 0.5 * log1p_nonnegative(strike_per_fight_source)
        )
        power_score = (
            0.5 * log1p_nonnegative(kd_per_fight_source)
            + 0.5 * log1p_nonnegative(kd_per_min_source * 60.0)
            + ko_rate_source.clip(lower=0.0)
            + 0.25 * power_index_source
        )
        grappling_score = (
            log1p_nonnegative(td_attempted_source)
            + log1p_nonnegative(td_landed_source)
            + log1p_nonnegative(ctrl_per_fight_source / 60.0)
            + log1p_nonnegative(sub_per_fight_source)
        )

        striker_column = f"pre_fight_{corner}_striker_score"
        grappler_column = f"pre_fight_{corner}_grappler_score"
        style_index_column = f"pre_fight_{corner}_style_index"
        striking_volume_column = f"pre_fight_{corner}_striking_volume_score"
        power_column = f"pre_fight_{corner}_power_score"
        grappling_component_column = f"pre_fight_{corner}_grappling_score"
        finish_score_column = f"pre_fight_{corner}_finish_score"
        vulnerability_column = f"pre_fight_{corner}_vulnerability_score"
        td_matchup_column = f"{corner}_td_offense_vs_{opponent_corner}_td_defense"
        grappling_pressure_column = f"{corner}_grappling_pressure_vs_{opponent_corner}_defense"

        enriched[striking_volume_column] = striking_volume_score
        enriched[power_column] = power_score
        enriched[grappling_component_column] = grappling_score
        enriched[striker_column] = 0.6 * striking_volume_score + 0.4 * power_score
        enriched[grappler_column] = grappling_score
        enriched[style_index_column] = enriched[grappler_column] - enriched[striker_column]
        enriched[finish_score_column] = (
            log1p_nonnegative(kd_per_fight_source)
            + ko_rate_source.clip(lower=0.0)
            + 0.5 * power_index_source
        )
        enriched[vulnerability_column] = (
            log1p_nonnegative(coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_kd_absorbed_per_fight"]))
            + coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_ko_loss_rate"]).clip(lower=0.0)
            + 0.5 * coalesce_numeric_columns(enriched, [f"pre_fight_{corner}_vulnerability_index"])
        )
        enriched[td_matchup_column] = td_landed_source.clip(lower=0.0) * (1.0 - opponent_td_defense_source.clip(lower=0.0, upper=1.0))
        enriched[grappling_pressure_column] = (
            (td_attempted_source.clip(lower=0.0) + sub_per_fight_source.clip(lower=0.0) + (ctrl_per_fight_source / 60.0).clip(lower=0.0))
            * (1.0 - opponent_td_defense_source.clip(lower=0.0, upper=1.0))
            * (0.5 + td_accuracy_source.clip(lower=0.0, upper=1.0))
        )
        created_columns.extend(
            [
                striking_volume_column,
                power_column,
                grappling_component_column,
                striker_column,
                grappler_column,
                style_index_column,
                finish_score_column,
                vulnerability_column,
                td_matchup_column,
                grappling_pressure_column,
            ]
        )

    interaction_specs = {
        "diff_striker_score": ("pre_fight_red_striker_score", "pre_fight_blue_striker_score"),
        "diff_grappler_score": ("pre_fight_red_grappler_score", "pre_fight_blue_grappler_score"),
        "diff_style_index": ("pre_fight_red_style_index", "pre_fight_blue_style_index"),
        "red_grappler_vs_blue_striker": ("pre_fight_red_grappler_score", "pre_fight_blue_striker_score"),
        "red_striker_vs_blue_grappler": ("pre_fight_red_striker_score", "pre_fight_blue_grappler_score"),
        "red_finish_vs_blue_durability": ("pre_fight_red_finish_score", "pre_fight_blue_vulnerability_score"),
        "blue_finish_vs_red_durability": ("pre_fight_blue_finish_score", "pre_fight_red_vulnerability_score"),
        "diff_td_offense_vs_td_defense": (
            "red_td_offense_vs_blue_td_defense",
            "blue_td_offense_vs_red_td_defense",
        ),
        "diff_grappling_pressure_vs_defense": (
            "red_grappling_pressure_vs_blue_defense",
            "blue_grappling_pressure_vs_red_defense",
        ),
    }
    for target_column, (left_column, right_column) in interaction_specs.items():
        if {left_column, right_column}.issubset(enriched.columns):
            left_values = pd.to_numeric(enriched[left_column], errors="coerce")
            right_values = pd.to_numeric(enriched[right_column], errors="coerce")
            if target_column.startswith("red_") or target_column.startswith("blue_"):
                enriched[target_column] = left_values * right_values
            else:
                enriched[target_column] = left_values - right_values
            created_columns.append(target_column)

    if {"red_grappler_vs_blue_striker", "red_striker_vs_blue_grappler"}.issubset(enriched.columns):
        enriched["diff_grappler_vs_striker_interaction"] = (
            pd.to_numeric(enriched["red_grappler_vs_blue_striker"], errors="coerce")
            - pd.to_numeric(enriched["red_striker_vs_blue_grappler"], errors="coerce")
        )
        created_columns.append("diff_grappler_vs_striker_interaction")

    if {"red_finish_vs_blue_durability", "blue_finish_vs_red_durability"}.issubset(enriched.columns):
        enriched["diff_finish_vs_durability"] = (
            pd.to_numeric(enriched["red_finish_vs_blue_durability"], errors="coerce")
            - pd.to_numeric(enriched["blue_finish_vs_red_durability"], errors="coerce")
        )
        created_columns.append("diff_finish_vs_durability")

    return enriched, created_columns


def add_temporal_hypothesis_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create layoff, cardio, and five-round experience features using only prior fights."""
    working = df.copy()
    created_columns: list[str] = []
    working["_temporal_hypothesis_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_temporal_hypothesis_order"], kind="mergesort")

    fighter_last_fight_date: dict[str, pd.Timestamp] = {}
    fighter_temporal_totals: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    base_columns = [
        "pre_fight_red_days_since_last_fight",
        "pre_fight_blue_days_since_last_fight",
        "pre_fight_red_short_turnaround",
        "pre_fight_blue_short_turnaround",
        "pre_fight_red_long_layoff",
        "pre_fight_blue_long_layoff",
        "pre_fight_red_avg_fight_duration_seconds",
        "pre_fight_blue_avg_fight_duration_seconds",
        "pre_fight_red_decision_rate",
        "pre_fight_blue_decision_rate",
        "pre_fight_red_late_finish_rate",
        "pre_fight_blue_late_finish_rate",
        "pre_fight_red_third_round_or_later_rate",
        "pre_fight_blue_third_round_or_later_rate",
        "pre_fight_red_five_round_experience",
        "pre_fight_blue_five_round_experience",
        "pre_fight_red_five_round_fights_completed",
        "pre_fight_blue_five_round_fights_completed",
        "pre_fight_red_past_5_round_scheduled_fights",
        "pre_fight_blue_past_5_round_scheduled_fights",
        "pre_fight_red_past_5_round_wins",
        "pre_fight_blue_past_5_round_wins",
        "pre_fight_red_past_5_round_losses",
        "pre_fight_blue_past_5_round_losses",
    ]
    for column in base_columns:
        working[column] = np.nan if "days_since_last_fight" in column else 0.0
        created_columns.append(column)

    for _, date_group in working.groupby("event_date", sort=False):
        for index, row in date_group.iterrows():
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
            for corner in ["red", "blue"]:
                fighter_name = row.get(f"{corner}_fighter_name")
                if pd.isna(fighter_name):
                    continue
                fighter_name = str(fighter_name)
                totals = fighter_temporal_totals[fighter_name]
                total_fights = float(totals["total_fights"])

                days_column = f"pre_fight_{corner}_days_since_last_fight"
                short_column = f"pre_fight_{corner}_short_turnaround"
                long_column = f"pre_fight_{corner}_long_layoff"
                last_date = fighter_last_fight_date.get(fighter_name)
                if last_date is not None and not pd.isna(event_date):
                    days_since_last = float(max((event_date - last_date).days, 0))
                    working.at[index, days_column] = days_since_last
                    working.at[index, short_column] = float(days_since_last < 100.0)
                    working.at[index, long_column] = float(days_since_last > 365.0)
                else:
                    working.at[index, days_column] = np.nan
                    working.at[index, short_column] = 0.0
                    working.at[index, long_column] = 0.0

                avg_duration_column = f"pre_fight_{corner}_avg_fight_duration_seconds"
                decision_rate_column = f"pre_fight_{corner}_decision_rate"
                late_finish_rate_column = f"pre_fight_{corner}_late_finish_rate"
                third_round_rate_column = f"pre_fight_{corner}_third_round_or_later_rate"
                five_round_experience_column = f"pre_fight_{corner}_five_round_experience"
                five_round_completed_column = f"pre_fight_{corner}_five_round_fights_completed"
                scheduled_column = f"pre_fight_{corner}_past_5_round_scheduled_fights"
                wins_column = f"pre_fight_{corner}_past_5_round_wins"
                losses_column = f"pre_fight_{corner}_past_5_round_losses"

                working.at[index, avg_duration_column] = (totals["total_duration_seconds"] / total_fights) if total_fights > 0 else 0.0
                working.at[index, decision_rate_column] = (totals["decision_fights"] / total_fights) if total_fights > 0 else 0.0
                working.at[index, late_finish_rate_column] = (totals["late_finishes"] / total_fights) if total_fights > 0 else 0.0
                working.at[index, third_round_rate_column] = (totals["third_round_or_later_fights"] / total_fights) if total_fights > 0 else 0.0
                working.at[index, five_round_experience_column] = totals["past_5_round_scheduled_fights"]
                working.at[index, five_round_completed_column] = totals["five_round_fights_completed"]
                working.at[index, scheduled_column] = totals["past_5_round_scheduled_fights"]
                working.at[index, wins_column] = totals["past_5_round_wins"]
                working.at[index, losses_column] = totals["past_5_round_losses"]

        for _, row in date_group.iterrows():
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
            fight_outcome = str(row.get("fight_outcome", "")).lower()
            round_number = pd.to_numeric(row.get("round"), errors="coerce")
            scheduled_rounds = parse_scheduled_rounds(row.get("time_format"))
            duration_seconds = get_numeric_value(row, "total_fight_duration_seconds")
            method_text = str(row.get("method", "")).lower()
            is_decision = float("decision" in method_text)
            reached_round_three = float(not pd.isna(round_number) and float(round_number) >= 3.0)
            late_finish = float(reached_round_three and not is_decision)
            five_round_scheduled = float(not pd.isna(scheduled_rounds) and float(scheduled_rounds) >= 5.0)
            five_round_completed = float(not pd.isna(round_number) and float(round_number) >= 4.0)

            for corner in ["red", "blue"]:
                fighter_name = row.get(f"{corner}_fighter_name")
                if pd.isna(fighter_name):
                    continue
                fighter_name = str(fighter_name)
                totals = fighter_temporal_totals[fighter_name]
                totals["total_fights"] += 1.0
                totals["total_duration_seconds"] += duration_seconds
                totals["decision_fights"] += is_decision
                totals["late_finishes"] += late_finish
                totals["third_round_or_later_fights"] += reached_round_three
                totals["past_5_round_scheduled_fights"] += five_round_scheduled
                totals["five_round_fights_completed"] += five_round_completed

                won = float(fight_outcome == f"{corner}_win")
                lost = float(fight_outcome in {"red_win", "blue_win"} and not won)
                totals["past_5_round_wins"] += five_round_scheduled * won
                totals["past_5_round_losses"] += five_round_scheduled * lost
                if not pd.isna(event_date):
                    fighter_last_fight_date[fighter_name] = event_date

    working = working.sort_values("_temporal_hypothesis_order", kind="mergesort").drop(columns=["_temporal_hypothesis_order"])

    diff_specs = {
        "diff_days_since_last_fight": ("pre_fight_red_days_since_last_fight", "pre_fight_blue_days_since_last_fight"),
        "diff_short_turnaround": ("pre_fight_red_short_turnaround", "pre_fight_blue_short_turnaround"),
        "diff_long_layoff": ("pre_fight_red_long_layoff", "pre_fight_blue_long_layoff"),
        "diff_avg_fight_duration_seconds": ("pre_fight_red_avg_fight_duration_seconds", "pre_fight_blue_avg_fight_duration_seconds"),
        "diff_decision_rate": ("pre_fight_red_decision_rate", "pre_fight_blue_decision_rate"),
        "diff_third_round_or_later_rate": ("pre_fight_red_third_round_or_later_rate", "pre_fight_blue_third_round_or_later_rate"),
        "diff_five_round_experience": ("pre_fight_red_five_round_experience", "pre_fight_blue_five_round_experience"),
        "diff_five_round_fights_completed": ("pre_fight_red_five_round_fights_completed", "pre_fight_blue_five_round_fights_completed"),
        "diff_past_5_round_scheduled_fights": ("pre_fight_red_past_5_round_scheduled_fights", "pre_fight_blue_past_5_round_scheduled_fights"),
        "diff_past_5_round_wins": ("pre_fight_red_past_5_round_wins", "pre_fight_blue_past_5_round_wins"),
        "diff_past_5_round_losses": ("pre_fight_red_past_5_round_losses", "pre_fight_blue_past_5_round_losses"),
    }
    for target, (red_column, blue_column) in diff_specs.items():
        if {red_column, blue_column}.issubset(working.columns):
            working[target] = pd.to_numeric(working[red_column], errors="coerce") - pd.to_numeric(
                working[blue_column], errors="coerce"
            )
            created_columns.append(target)

    return working, created_columns


def add_momentum_features(df: pd.DataFrame, recent_n: int = DEFAULT_LAST_N_RECENCY) -> tuple[pd.DataFrame, list[str]]:
    """Compare recent last-N form against leakage-safe career pre-fight baselines."""
    enriched = df.copy()
    created_columns: list[str] = []

    for corner in ["red", "blue"]:
        metric_specs = {
            "momentum_win_pct": (
                f"recent_{recent_n}_{corner}_win_pct",
                f"pre_fight_{corner}_win_pct",
            ),
            "momentum_sig_strikes_per_min": (
                f"recent_{recent_n}_{corner}_sig_strikes_landed_per_min",
                f"pre_fight_{corner}_sig_strikes_landed_per_min",
            ),
            "momentum_td_landed_per_fight": (
                f"recent_{recent_n}_{corner}_td_landed_per_fight",
                f"pre_fight_{corner}_td_landed_per_fight",
            ),
            "momentum_kd_absorbed_per_fight": (
                f"recent_{recent_n}_{corner}_kd_absorbed_per_fight",
                f"pre_fight_{corner}_kd_absorbed_per_fight",
            ),
        }
        for feature_name, (recent_column, career_column) in metric_specs.items():
            if {recent_column, career_column}.issubset(enriched.columns):
                target = f"pre_fight_{corner}_{feature_name}"
                enriched[target] = pd.to_numeric(enriched[recent_column], errors="coerce") - pd.to_numeric(
                    enriched[career_column], errors="coerce"
                )
                created_columns.append(target)

    diff_specs = {
        "diff_momentum_win_pct": (
            "pre_fight_red_momentum_win_pct",
            "pre_fight_blue_momentum_win_pct",
        ),
        "diff_momentum_sig_strikes_per_min": (
            "pre_fight_red_momentum_sig_strikes_per_min",
            "pre_fight_blue_momentum_sig_strikes_per_min",
        ),
        "diff_momentum_td_landed_per_fight": (
            "pre_fight_red_momentum_td_landed_per_fight",
            "pre_fight_blue_momentum_td_landed_per_fight",
        ),
        "diff_momentum_kd_absorbed_per_fight": (
            "pre_fight_red_momentum_kd_absorbed_per_fight",
            "pre_fight_blue_momentum_kd_absorbed_per_fight",
        ),
    }
    for target, (red_column, blue_column) in diff_specs.items():
        if {red_column, blue_column}.issubset(enriched.columns):
            enriched[target] = pd.to_numeric(enriched[red_column], errors="coerce") - pd.to_numeric(
                enriched[blue_column], errors="coerce"
            )
            created_columns.append(target)

    return enriched, created_columns


def print_hypothesis_feature_diagnostics(df: pd.DataFrame, feature_columns: list[str], recent_n: int) -> None:
    """Print compact sanity checks and previews for the new human-hypothesis features."""
    preview_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        "pre_fight_red_striker_score",
        "pre_fight_blue_striker_score",
        "pre_fight_red_grappler_score",
        "pre_fight_blue_grappler_score",
        "diff_style_index",
        "red_finish_vs_blue_durability",
        "blue_finish_vs_red_durability",
        "diff_finish_vs_durability",
        "pre_fight_red_days_since_last_fight",
        "pre_fight_blue_days_since_last_fight",
        "diff_days_since_last_fight",
        f"pre_fight_red_momentum_win_pct",
        f"pre_fight_blue_momentum_win_pct",
        "diff_momentum_win_pct",
        "diff_avg_fight_duration_seconds",
        "diff_past_5_round_scheduled_fights",
        "diff_td_offense_vs_td_defense",
    ]
    available_preview = [column for column in preview_columns if column in df.columns]

    red_first = df.get("pre_fight_red_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    blue_first = df.get("pre_fight_blue_total_fights", pd.Series(dtype="float64")).fillna(0).eq(0)
    red_layoff_zero = int(pd.to_numeric(df.loc[red_first, "pre_fight_red_days_since_last_fight"], errors="coerce").fillna(np.nan).eq(0).sum()) if "pre_fight_red_days_since_last_fight" in df.columns else 0
    blue_layoff_zero = int(pd.to_numeric(df.loc[blue_first, "pre_fight_blue_days_since_last_fight"], errors="coerce").fillna(np.nan).eq(0).sum()) if "pre_fight_blue_days_since_last_fight" in df.columns else 0
    recent_column = f"recent_{recent_n}_red_win_pct"
    momentum_column = "pre_fight_red_momentum_win_pct"
    momentum_example = []
    if {recent_column, "pre_fight_red_win_pct", momentum_column}.issubset(df.columns):
        sample = df[[recent_column, "pre_fight_red_win_pct", momentum_column]].head(5).copy()
        sample["reconstructed_momentum"] = pd.to_numeric(sample[recent_column], errors="coerce") - pd.to_numeric(sample["pre_fight_red_win_pct"], errors="coerce")
        momentum_example = sample.round(4).to_dict(orient="records")

    print("New human-hypothesis columns:")
    print(list(dict.fromkeys(feature_columns)))
    print("Human-hypothesis sanity checks:")
    print(f"- red first-fight layoff values incorrectly treated as zero-day layoffs: {red_layoff_zero}")
    print(f"- blue first-fight layoff values incorrectly treated as zero-day layoffs: {blue_layoff_zero}")
    print(f"- sample momentum reconstruction checks: {momentum_example}")
    print("Human-hypothesis preview (first 5 rows):")
    print(df[available_preview].head())

def create_cumulative_prefight_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create cumulative fighter statistics using only prior fights.

    The function iterates in chronological order. For each row, it writes the
    fighter's current cumulative totals before updating those totals with the
    current fight's result and stats.
    """
    working = df.copy()
    working["_original_order"] = np.arange(len(working))
    working = working.sort_values(["event_date", "_original_order"], kind="mergesort")

    fighter_totals: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    created_columns: list[str] = []

    base_feature_names = [
        "total_fights",
        "total_wins",
        *OUTCOME_CUMULATIVE_COLUMNS,
        *CUMULATIVE_SOURCE_COLUMNS.keys(),
        *ABSORBED_CUMULATIVE_COLUMNS.keys(),
    ]

    for corner in ["red", "blue"]:
        for feature_name in base_feature_names:
            column = f"pre_fight_{corner}_{feature_name}"
            working[column] = 0.0
            created_columns.append(column)

    for _, date_group in working.groupby("event_date", sort=False):
        # Snapshot before updating the date group so same-day fights cannot leak.
        for index, row in date_group.iterrows():
            red_name = row.get("red_fighter_name")
            blue_name = row.get("blue_fighter_name")

            for corner, fighter_name in [("red", red_name), ("blue", blue_name)]:
                snapshot = fighter_snapshot(fighter_totals[str(fighter_name)])
                for feature_name, value in snapshot.items():
                    working.at[index, f"pre_fight_{corner}_{feature_name}"] = value

        for _, row in date_group.iterrows():
            red_name = row.get("red_fighter_name")
            blue_name = row.get("blue_fighter_name")
            fight_outcome = str(row.get("fight_outcome", "")).lower()
            red_won = int(fight_outcome == "red_win")
            blue_won = int(fight_outcome == "blue_win")

            for corner, fighter_name, won in [("red", red_name, red_won), ("blue", blue_name, blue_won)]:
                totals = fighter_totals[str(fighter_name)]
                totals["total_fights"] += 1
                totals["total_wins"] += won
                lost = int(fight_outcome in {"red_win", "blue_win"} and not won)
                totals["total_losses"] += lost

                for outcome_name, outcome_value in method_win_flags(row.get("method"), won).items():
                    totals[outcome_name] += outcome_value
                for outcome_name, outcome_value in method_loss_flags(row.get("method"), lost).items():
                    totals[outcome_name] += outcome_value

                for output_name, source_suffix in CUMULATIVE_SOURCE_COLUMNS.items():
                    source_column = f"{corner}_{source_suffix}"
                    totals[output_name] += get_numeric_value(row, source_column)

                for output_name, corner_map in ABSORBED_CUMULATIVE_COLUMNS.items():
                    totals[output_name] += get_numeric_value(row, corner_map[corner])

    working = working.sort_values("_original_order", kind="mergesort").drop(columns=["_original_order"])
    working, efficiency_columns = add_prefight_efficiency_columns(working)
    created_columns.extend(efficiency_columns)
    return working, created_columns


def print_prefight_feature_diagnostics(df: pd.DataFrame, prefight_columns: list[str]) -> None:
    """Print the pre-fight columns added by this run plus a compact preview."""
    unique_prefight_columns = list(dict.fromkeys(prefight_columns))
    example_columns = [
        "red_fighter_name",
        "blue_fighter_name",
        "event_date",
        "pre_fight_red_total_fights",
        "pre_fight_red_total_wins",
        "pre_fight_red_total_losses",
        "pre_fight_red_total_ko_wins",
        "pre_fight_red_total_submission_wins",
        "pre_fight_red_total_decision_wins",
        "pre_fight_red_total_sig_strikes_head_landed",
        "pre_fight_red_total_sig_strikes_head_attempted",
        "pre_fight_red_sig_strike_accuracy",
        "pre_fight_red_td_accuracy",
        "pre_fight_red_win_pct",
        "pre_fight_red_sig_strikes_landed_per_fight",
        "pre_fight_red_sig_strikes_landed_per_min",
        "pre_fight_blue_total_fights",
        "pre_fight_blue_total_wins",
        "pre_fight_blue_total_losses",
        "pre_fight_blue_total_ko_wins",
        "pre_fight_blue_total_submission_wins",
        "pre_fight_blue_total_decision_wins",
        "pre_fight_blue_total_sig_strikes_head_landed",
        "pre_fight_blue_total_sig_strikes_head_attempted",
        "pre_fight_blue_sig_strike_accuracy",
        "pre_fight_blue_td_accuracy",
        "pre_fight_blue_win_pct",
        "pre_fight_blue_sig_strikes_landed_per_fight",
        "pre_fight_blue_sig_strikes_landed_per_min",
    ]
    available_examples = [column for column in example_columns if column in df.columns]

    print("Newly added pre-fight columns:")
    print(unique_prefight_columns)
    print("Pre-fight feature examples (first 5 rows):")
    print(df[available_examples].head())


def print_prefight_context_diagnostics(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Print new context features and a compact chronology sanity check preview."""
    unique_columns = list(dict.fromkeys(feature_columns))
    preview_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        "weight_class_clean",
        "pre_fight_red_win_streak",
        "pre_fight_blue_win_streak",
        "pre_fight_red_loss_streak",
        "pre_fight_blue_loss_streak",
        "pre_fight_red_avg_past_opponent_win_pct",
        "pre_fight_blue_avg_past_opponent_win_pct",
        "pre_fight_red_years_in_ufc",
        "pre_fight_blue_years_in_ufc",
        "pre_fight_red_fights_per_year",
        "pre_fight_blue_fights_per_year",
        "diff_win_streak",
        "diff_avg_past_opponent_win_pct",
        "diff_years_in_ufc",
    ]
    available_preview = [column for column in preview_columns if column in df.columns]
    print("New pre-fight context columns:")
    print(unique_columns)
    print("Context feature preview (first 5 rows):")
    print(df[available_preview].head())


def print_prefight_sanity_checks(df: pd.DataFrame) -> None:
    """Run a few lightweight checks to confirm chronology-safe features look sensible."""
    checks: list[tuple[str, int]] = []
    if {"pre_fight_red_total_fights", "pre_fight_red_win_streak", "pre_fight_red_loss_streak"}.issubset(df.columns):
        first_red = df["pre_fight_red_total_fights"].fillna(0).eq(0)
        invalid_red = int(
            ((df.loc[first_red, "pre_fight_red_win_streak"].fillna(0) != 0) | (df.loc[first_red, "pre_fight_red_loss_streak"].fillna(0) != 0)).sum()
        )
        checks.append(("red first-fight streak mismatches", invalid_red))
    if {"pre_fight_blue_total_fights", "pre_fight_blue_win_streak", "pre_fight_blue_loss_streak"}.issubset(df.columns):
        first_blue = df["pre_fight_blue_total_fights"].fillna(0).eq(0)
        invalid_blue = int(
            ((df.loc[first_blue, "pre_fight_blue_win_streak"].fillna(0) != 0) | (df.loc[first_blue, "pre_fight_blue_loss_streak"].fillna(0) != 0)).sum()
        )
        checks.append(("blue first-fight streak mismatches", invalid_blue))
    if {"pre_fight_red_total_wins", "pre_fight_red_total_fights"}.issubset(df.columns):
        red_leakage = int((df["pre_fight_red_total_wins"].fillna(0) > df["pre_fight_red_total_fights"].fillna(0)).sum())
        checks.append(("red win totals exceeding fights", red_leakage))
    if {"pre_fight_blue_total_wins", "pre_fight_blue_total_fights"}.issubset(df.columns):
        blue_leakage = int((df["pre_fight_blue_total_wins"].fillna(0) > df["pre_fight_blue_total_fights"].fillna(0)).sum())
        checks.append(("blue win totals exceeding fights", blue_leakage))

    print("Pre-fight sanity checks:")
    for label, value in checks:
        print(f"- {label}: {value}")

    fighter_rows: list[pd.DataFrame] = []
    stacked = pd.concat(
        [
            df[["red_fighter_name"]].rename(columns={"red_fighter_name": "fighter_name"}),
            df[["blue_fighter_name"]].rename(columns={"blue_fighter_name": "fighter_name"}),
        ],
        ignore_index=True,
    )
    sample_fighters = stacked["fighter_name"].value_counts().head(3).index.tolist()
    example_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        "pre_fight_red_total_fights",
        "pre_fight_blue_total_fights",
        "pre_fight_red_win_streak",
        "pre_fight_blue_win_streak",
        "pre_fight_red_avg_past_opponent_win_pct",
        "pre_fight_blue_avg_past_opponent_win_pct",
    ]
    available_columns = [column for column in example_columns if column in df.columns]
    for fighter_name in sample_fighters:
        fighter_rows.append(
            df.loc[
                (df["red_fighter_name"] == fighter_name) | (df["blue_fighter_name"] == fighter_name),
                available_columns,
            ]
            .sort_values("event_date")
            .head(5)
        )
    if fighter_rows:
        print("Example fighter chronology checks:")
        print(pd.concat(fighter_rows).head(15))


def print_rate_and_difference_diagnostics(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Print created rate/efficiency/difference columns with a compact preview."""
    unique_columns = list(dict.fromkeys(feature_columns))
    example_columns = [
        "red_fighter_name",
        "blue_fighter_name",
        "red_fighter_slpm",
        "blue_fighter_slpm",
        "slpm_diff",
        "red_fighter_td_def",
        "blue_fighter_td_def",
        "td_def_diff",
        "red_fighter_reach",
        "blue_fighter_reach",
        "reach_diff",
        "red_fighter_sig_strikes_landed_per_minute",
        "blue_fighter_sig_strikes_landed_per_minute",
        "sig_strikes_landed_per_minute_diff",
        "red_fighter_sig_strike_accuracy",
        "blue_fighter_sig_strike_accuracy",
        "sig_strike_accuracy_diff",
        "pre_fight_red_sig_strikes_landed_per_minute",
        "pre_fight_blue_sig_strikes_landed_per_minute",
        "pre_fight_sig_strikes_landed_per_minute_diff",
        "pre_fight_red_win_rate",
        "pre_fight_blue_win_rate",
        "pre_fight_win_rate_diff",
    ]
    available_examples = [column for column in example_columns if column in df.columns]

    print("New rate/efficiency and difference columns:")
    print(unique_columns)
    print("Rate/difference examples (first 5 rows):")
    print(df[available_examples].head())


def print_power_and_durability_diagnostics(
    df: pd.DataFrame,
    feature_columns: list[str],
    weights: dict[str, float],
) -> None:
    """Print the newly created power and durability features plus a small preview."""
    unique_columns = list(dict.fromkeys(feature_columns))
    preview_columns = [
        "event_date",
        "red_fighter_name",
        "blue_fighter_name",
        "pre_fight_red_power_index",
        "pre_fight_blue_power_index",
        "pre_fight_red_durability_index",
        "pre_fight_blue_durability_index",
        "pre_fight_red_opponent_adjusted_power",
        "pre_fight_blue_opponent_adjusted_power",
        "diff_power_index",
        "diff_durability_index",
        "diff_opponent_adjusted_power",
    ]
    available_preview = [column for column in preview_columns if column in df.columns]

    print("Strike-type weights used for weighted power features:")
    print(weights)
    print("New opponent-adjusted power and durability columns:")
    print(unique_columns)
    print("Power/durability preview (first 5 rows):")
    print(df[available_preview].head())


def load_fighter_statistics(path: Path) -> pd.DataFrame | None:
    """Load raw fighter statistics if available, otherwise return None with a warning."""
    if not path.exists():
        print(f"Raw fighter statistics file not found at {path}; static attributes will be empty.")
        return None

    return load_csv_with_detected_delimiter(path, "raw fighter statistics")


# Backward-compatible name for earlier notebook imports.
load_fighter_details = load_fighter_statistics


def build_combined_dataset(
    fight_path: Path = DEFAULT_FIGHT_PATH,
    fighter_statistics_path: Path = DEFAULT_FIGHTER_STATISTICS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    elo_k_factor: float = DEFAULT_ELO_K_FACTOR,
    elo_initial_rating: float = DEFAULT_ELO_INITIAL_RATING,
    recency_last_n: int = DEFAULT_LAST_N_RECENCY,
    recency_half_life_days: float = DEFAULT_EXP_DECAY_HALF_LIFE_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full cleaning, merging, and pre-fight feature pipeline."""
    fights_raw = load_csv_with_detected_delimiter(fight_path, "fight data")
    fighter_statistics_raw = load_fighter_statistics(fighter_statistics_path)

    created_columns: list[str] = []

    fights = clean_fight_data(fights_raw)
    created_columns.extend(
        [
            column
            for column in ["red_win", "red_fighter_name_key", "blue_fighter_name_key"]
            if column in fights.columns
        ]
    )
    fights, stat_columns = parse_stat_columns(fights)
    created_columns.extend(stat_columns)

    fights, time_columns = convert_time_columns(fights)
    created_columns.extend(time_columns)

    fighter_statistics = clean_fighter_statistics(fighter_statistics_raw)
    fights, attribute_columns = merge_fighter_attributes(fights, fighter_statistics)
    created_columns.extend(attribute_columns)

    fights, prefight_columns = create_cumulative_prefight_stats(fights)
    created_columns.extend(prefight_columns)
    print_prefight_feature_diagnostics(fights, prefight_columns)

    fights, derived_prefight_columns = add_prefight_derived_features(fights)
    created_columns.extend(derived_prefight_columns)

    fights, rate_columns = add_rate_and_efficiency_features(fights)
    created_columns.extend(rate_columns)

    fights, power_columns, strike_type_weights = add_power_and_durability_features(fights)
    created_columns.extend(power_columns)
    print_power_and_durability_diagnostics(fights, power_columns, strike_type_weights)

    fights, context_columns = add_prefight_context_features(fights)
    created_columns.extend(context_columns)
    print_prefight_context_diagnostics(fights, context_columns)

    fights, last_n_recency_columns = build_last_n_recency_features(
        fights,
        n_last=recency_last_n,
    )
    created_columns.extend(last_n_recency_columns)

    fights, exp_decay_recency_columns = build_exp_decay_recency_features(
        fights,
        half_life_days=recency_half_life_days,
    )
    created_columns.extend(exp_decay_recency_columns)
    print_recency_feature_diagnostics(
        fights,
        last_n_recency_columns,
        exp_decay_recency_columns,
        n_last=recency_last_n,
        half_life_days=recency_half_life_days,
    )

    fights, temporal_hypothesis_columns = add_temporal_hypothesis_features(fights)
    created_columns.extend(temporal_hypothesis_columns)

    fights, style_hypothesis_columns = add_style_and_matchup_features(fights)
    created_columns.extend(style_hypothesis_columns)

    fights, momentum_columns = add_momentum_features(fights, recent_n=recency_last_n)
    created_columns.extend(momentum_columns)
    print_hypothesis_feature_diagnostics(
        fights,
        [*temporal_hypothesis_columns, *style_hypothesis_columns, *momentum_columns],
        recent_n=recency_last_n,
    )

    fights, elo_columns = build_elo_features(
        fights,
        k_factor=elo_k_factor,
        initial_rating=elo_initial_rating,
    )
    created_columns.extend(elo_columns)
    print_elo_diagnostics(fights, elo_columns, initial_rating=elo_initial_rating)

    fights, difference_columns = add_difference_features(fights)
    created_columns.extend(difference_columns)

    fights, alias_difference_columns = add_explicit_difference_aliases(fights)
    created_columns.extend(alias_difference_columns)
    print_rate_and_difference_diagnostics(
        fights,
        [*derived_prefight_columns, *rate_columns, *difference_columns, *alias_difference_columns],
    )
    print_prefight_sanity_checks(fights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fights.to_csv(output_path, index=False)

    created_columns = list(dict.fromkeys(created_columns))
    print(f"Final dataset shape: {fights.shape}")
    print("Missing values by column for the first 20 columns:")
    print(fights.iloc[:, :20].isna().sum())
    print("Newly created columns:")
    print(created_columns)
    print(f"Saved combined dataset to: {output_path}")

    return fights, created_columns


def main() -> None:
    """Command-line entry point for building the combined dataset."""
    build_combined_dataset()


if __name__ == "__main__":
    main()
