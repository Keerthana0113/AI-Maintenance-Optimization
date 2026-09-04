from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# DATASET PATH
# ============================================================

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simulated_iiot_dataset.csv"
)


# ============================================================
# SENSOR FEATURES
# ============================================================

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rotation_speed",
    "voltage",
    "current",
    "oil_level",
    "load",
    "motor_temperature",
    "gearbox_temperature",
    "sound_level",
    "fan_speed",
    "reactive_power",
    "active_power",
]


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "machine_failure"


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """Load the raw IIoT dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the dataset structure."""

    required_columns = (
        ["timestamp"]
        + SENSOR_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.duplicated().any():
        print("Warning: duplicate rows detected.")

    if df[SENSOR_COLUMNS].isnull().any().any():
        raise ValueError(
            "Missing values found in sensor columns."
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Missing values found in target column."
        )

    invalid_targets = (
        set(df[TARGET_COLUMN].unique()) - {0, 1}
    )

    if invalid_targets:
        raise ValueError(
            f"Invalid target values: {invalid_targets}"
        )


# ============================================================
# GET SENSOR COLUMNS
# ============================================================

def get_sensor_columns() -> list[str]:
    """Return sensor feature names."""

    return SENSOR_COLUMNS.copy()