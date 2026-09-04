from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from ai_maintenance_optimization.data.loader import (
    SENSOR_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data():
    """Load training and testing datasets."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing data not found: {TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained classification model."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ).tolist(),
    }

    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_models():

    print("=" * 60)
    print("AI MAINTENANCE OPTIMIZATION - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    train_df, test_df = load_processed_data()

    X_train = train_df[SENSOR_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[SENSOR_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\nTraining data:", X_train.shape)
    print("Testing data: ", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Logistic Regression...")
    print("-" * 60)

    logistic_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test_scaled,
        y_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Random Forest...")
    print("-" * 60)

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\nLOGISTIC REGRESSION")

    for metric, value in logistic_metrics.items():
        print(f"{metric}: {value}")

    print("\nRANDOM FOREST")

    for metric, value in random_forest_metrics.items():
        print(f"{metric}: {value}")

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    # Failure recall is the primary metric.
    # F1 score is used as the tie-breaker.

    if (
        random_forest_metrics["recall"],
        random_forest_metrics["f1_score"],
    ) > (
        logistic_metrics["recall"],
        logistic_metrics["f1_score"],
    ):
        best_model = random_forest_model
        best_metrics = random_forest_metrics
        best_model_name = "Random Forest"
        uses_scaler = False

    else:
        best_model = logistic_model
        best_metrics = logistic_metrics
        best_model_name = "Logistic Regression"
        uses_scaler = True

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"\nSelected model: {best_model_name}")
    print(f"Failure recall: {best_metrics['recall']:.4f}")
    print(f"F1 score:       {best_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:        {best_metrics['roc_auc']:.4f}")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODELS_DIR / "model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    metrics_data = {
        "selected_model": best_model_name,
        "uses_scaler": uses_scaler,
        "logistic_regression": logistic_metrics,
        "random_forest": random_forest_metrics,
        "selected_model_metrics": best_metrics,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("MODEL SAVING COMPLETE")
    print("=" * 60)

    print(f"\nModel:   {model_path}")
    print(f"Scaler:  {scaler_path}")
    print(f"Metrics: {metrics_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_models()