
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Class mapping confirmed by Muqeet
BENIGN = "Benign"
RANSOMWARE = "Ransomware"
LABELS = [BENIGN, RANSOMWARE]

REQUIRED_COLUMNS = [
    "execution_id",
    "actual_label",
    "predicted_label",
]


def normalize_label(value):
    """Standardise labels from the prediction file."""

    if pd.isna(value):
        raise ValueError("Missing label detected.")

    label = str(value).strip().lower()

    if label == "benign":
        return BENIGN

    if label == "ransomware":
        return RANSOMWARE

    raise ValueError(f"Invalid class label: {value}")


def load_predictions(file_path):
    """Load prediction records from CSV or JSON."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            data = data.get("predictions")

        if not isinstance(data, list):
            raise ValueError(
                "JSON must contain a list of prediction records."
            )

        df = pd.DataFrame(data)

    else:
        raise ValueError("Please provide a CSV or JSON file.")

    return df


def evaluate_predictions(df):
    """Calculate execution-level classification metrics."""

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("No prediction records found.")

    df = df.copy()

    if df["execution_id"].isna().any():
        raise ValueError("Missing execution ID detected.")

    if df["execution_id"].duplicated().any():
        raise ValueError(
            "Duplicate execution IDs found. "
            "Expected one prediction per execution."
        )

    if df["actual_label"].isna().any():
        raise ValueError(
            "Actual labels are required for offline evaluation."
        )

    df["actual_label"] = df["actual_label"].apply(normalize_label)
    df["predicted_label"] = df["predicted_label"].apply(normalize_label)

    y_true = df["actual_label"]
    y_pred = df["predicted_label"]

    # Confusion matrix: rows = actual, columns = predicted
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=RANSOMWARE,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=RANSOMWARE,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=RANSOMWARE,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None

    results = {
        "evaluation_unit": "execution",
        "positive_class": RANSOMWARE,
        "sample_count": int(len(df)),

        "class_support": {
            "Benign": int((y_true == BENIGN).sum()),
            "Ransomware": int((y_true == RANSOMWARE).sum())
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },

        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "false_positive_rate": (
                float(fpr) if fpr is not None else None
            ),
            "false_negative_rate": (
                float(fnr) if fnr is not None else None
            )
        }
    }

    # Calculate ROC-AUC and PR-AUC when valid probabilities exist.
    probability_columns = [
        "benign_probability",
        "ransomware_probability"
    ]

    if all(col in df.columns for col in probability_columns):

        probabilities = df[probability_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if probabilities.isna().any().any():
            raise ValueError("Invalid probability value detected.")

        if (
            (probabilities < 0).any().any()
            or (probabilities > 1).any().any()
        ):
            raise ValueError("Probabilities must be between 0 and 1.")

        if not (
            (probabilities.sum(axis=1) - 1).abs() <= 0.01
        ).all():
            raise ValueError(
                "Class probabilities must sum to approximately 1."
            )

        if y_true.nunique() == 2:
            y_binary = (y_true == RANSOMWARE).astype(int)

            ransomware_scores = probabilities[
                "ransomware_probability"
            ]

            results["metrics"]["roc_auc"] = float(
                roc_auc_score(y_binary, ransomware_scores)
            )

            results["metrics"]["pr_auc"] = float(
                average_precision_score(
                    y_binary,
                    ransomware_scores
                )
            )

        else:
            results["metrics"]["roc_auc"] = None
            results["metrics"]["pr_auc"] = None

    else:
        results["metrics"]["roc_auc"] = None
        results["metrics"]["pr_auc"] = None

    return results


def main():

    if len(sys.argv) != 2:
        print("Usage: python evaluation.py predictions.csv")
        print("   or: python evaluation.py predictions.json")
        return

    predictions = load_predictions(sys.argv[1])

    results = evaluate_predictions(predictions)

    output_file = "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n===== MODEL EVALUATION RESULTS =====")
    print(json.dumps(results, indent=4))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
