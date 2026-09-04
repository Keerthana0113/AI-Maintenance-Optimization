from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ai_maintenance_optimization.data.loader import (
    TARGET_COLUMN,
    get_sensor_columns,
    load_raw_data,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def prepare_data():
    """
    Load the raw dataset, validate it,
    convert timestamps, and create train/test datasets.
    """

    # 1. Load raw data
    df = load_raw_data()

    # 2. Validate dataset
    validate_dataset(df)

    # 3. Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isnull().any():
        raise ValueError("Invalid timestamp values found.")

    # 4. Select sensor features
    sensor_columns = get_sensor_columns()

    X = df[sensor_columns]
    y = df[TARGET_COLUMN]

    # 5. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 6. Create training DataFrame
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    # 7. Create testing DataFrame
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    # 8. Create processed directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. Save processed datasets
    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    return train_df, test_df


if __name__ == "__main__":

    train_df, test_df = prepare_data()

    print("=" * 60)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Testing data shape:  {test_df.shape}")

    print("\nTraining target distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nTesting target distribution:")
    print(test_df[TARGET_COLUMN].value_counts())

    print("\nFiles created:")
    print(TRAIN_PATH)
    print(TEST_PATH)