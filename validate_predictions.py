
import sys
import pandas as pd

from evaluation import (
    load_predictions,
    normalize_label,
    REQUIRED_COLUMNS,
)


PROBABILITY_COLUMNS = [
    "benign_probability",
    "ransomware_probability",
]

OPTIONAL_METADATA = [
    "experiment_name",
    "adaptation_stage",
    "category",
    "hardware_config",
]


def validate_predictions(file_path, offline=True):
    """Validate prediction records before model evaluation."""

    df = load_predictions(file_path)

    if df.empty:
        raise ValueError("Prediction file is empty.")

    # Check required columns.
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check execution IDs.
    if df["execution_id"].isna().any():
        raise ValueError("Missing execution ID.")

    df["execution_id"] = df["execution_id"].astype(str).str.strip()

    if (df["execution_id"] == "").any():
        raise ValueError("Empty execution ID detected.")

    if df["execution_id"].duplicated().any():
        raise ValueError("Duplicate execution IDs detected.")

    # Check labels.
    df["predicted_label"] = df["predicted_label"].apply(
        normalize_label
    )

    if offline:
        if df["actual_label"].isna().any():
            raise ValueError(
                "Actual labels are required for offline evaluation."
            )

        df["actual_label"] = df["actual_label"].apply(
            normalize_label
        )

    # Probability fields must either both exist or both be absent.
    present = [
        col for col in PROBABILITY_COLUMNS
        if col in df.columns
    ]

    if len(present) == 1:
        raise ValueError(
            "Both benign_probability and "
            "ransomware_probability must be provided together."
        )

    if len(present) == 2:
        probabilities = df[PROBABILITY_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if probabilities.isna().any().any():
            raise ValueError("Missing or invalid probability values.")

        if (
            (probabilities < 0).any().any()
            or (probabilities > 1).any().any()
        ):
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        if not (
            (probabilities.sum(axis=1) - 1).abs() <= 0.01
        ).all():
            raise ValueError(
                "Class probabilities must sum approximately to 1."
            )

        # Check argmax consistency only when explicitly requested
        # for the baseline model.
        if "--check-argmax" in sys.argv:
            expected = probabilities.idxmax(axis=1).map({
                "benign_probability": "Benign",
                "ransomware_probability": "Ransomware"
            })

            if not (
                expected.values == df["predicted_label"].values
            ).all():
                raise ValueError(
                    "Predicted labels do not match probability argmax."
                )

    # Check confidence range if supplied.
    if "confidence" in df.columns:
        confidence = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        if confidence.isna().any():
            raise ValueError("Invalid confidence values.")

        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("Confidence must be between 0 and 1.")

    print("\n===== PREDICTION VALIDATION PASSED =====")
    print(f"Prediction records: {len(df)}")
    print(f"Unique executions: {df['execution_id'].nunique()}")
    print(f"Offline evaluation: {offline}")
    print(f"Probabilities available: {len(present) == 2}")

    for column in OPTIONAL_METADATA:
        if column in df.columns:
            print(f"{column}: available")

    return df


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python3 validate_predictions.py "
            "predictions.csv"
        )
        sys.exit(1)

    try:
        validate_predictions(sys.argv[1])

    except (ValueError, FileNotFoundError) as error:
        print(f"\nVALIDATION FAILED: {error}")
        sys.exit(1)
