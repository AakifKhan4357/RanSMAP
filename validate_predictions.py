
import sys
import pandas as pd

from evaluation import load_predictions, normalize_label


BENIGN = "Benign"
RANSOMWARE = "Ransomware"
TOLERANCE = 1e-6

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


def validate_predictions(file_path, offline=True, check_argmax=True):
    """Validate predictions using the current argmax baseline contract."""

    df = load_predictions(file_path)

    if df.empty:
        raise ValueError("Prediction file is empty.")

    required_columns = [
        "execution_id",
        "predicted_label",
    ]

    if offline:
        required_columns.append("actual_label")

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Execution ID checks
    if df["execution_id"].isna().any():
        raise ValueError("Missing execution ID.")

    df["execution_id"] = (
        df["execution_id"].astype(str).str.strip()
    )

    if (df["execution_id"] == "").any():
        raise ValueError("Empty execution ID detected.")

    if df["execution_id"].duplicated().any():
        raise ValueError("Duplicate execution IDs detected.")

    # Label checks
    df["predicted_label"] = df["predicted_label"].apply(
        normalize_label
    )

    if offline:
        if df["actual_label"].isna().any():
            raise ValueError("Missing actual labels.")

        df["actual_label"] = df["actual_label"].apply(
            normalize_label
        )

    # Both probability columns must be supplied together.
    present = [
        col for col in PROBABILITY_COLUMNS
        if col in df.columns
    ]

    if len(present) != 2:
        raise ValueError(
            "Both probability columns are required "
            "for the current argmax baseline."
        )

    probabilities = df[PROBABILITY_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if probabilities.isna().any().any():
        raise ValueError("Invalid probability values.")

    if (
        (probabilities < 0).any().any()
        or (probabilities > 1).any().any()
    ):
        raise ValueError("Probabilities must be between 0 and 1.")

    benign = probabilities["benign_probability"]
    ransomware = probabilities["ransomware_probability"]

    # Probability sum check
    if not (
        (benign + ransomware - 1.0).abs() <= TOLERANCE
    ).all():
        raise ValueError(
            "Class probabilities must sum to approximately 1.0."
        )

    # Confidence must match the maximum probability.
    if "confidence" not in df.columns:
        raise ValueError("Missing confidence column.")

    confidence = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    if confidence.isna().any():
        raise ValueError("Invalid confidence values.")

    if (
        (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError("Confidence must be between 0 and 1.")

    expected_confidence = probabilities.max(axis=1)

    if not (
        (confidence - expected_confidence).abs() <= TOLERANCE
    ).all():
        raise ValueError(
            "Confidence must equal the maximum class probability."
        )

    # Current baseline uses argmax.
    if check_argmax:
        expected_labels = [
            BENIGN if b >= r else RANSOMWARE
            for b, r in zip(benign, ransomware)
        ]

        if expected_labels != df["predicted_label"].tolist():
            raise ValueError(
                "Predicted labels do not match argmax probabilities."
            )

    print("\n===== PREDICTION VALIDATION PASSED =====")
    print(f"Prediction records: {len(df)}")
    print(f"Unique executions: {df['execution_id'].nunique()}")
    print(f"Offline evaluation: {offline}")
    print("Probability sum check: passed")
    print("Confidence consistency check: passed")
    print(f"Argmax consistency check: {check_argmax}")

    for column in OPTIONAL_METADATA:
        if column in df.columns:
            print(f"{column}: available")

    return df


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python3 validate_predictions.py predictions.csv"
        )
        sys.exit(1)

    try:
        validate_predictions(sys.argv[1])

    except (ValueError, FileNotFoundError) as error:
        print(f"\nVALIDATION FAILED: {error}")
        sys.exit(1)
