
import sys
import pandas as pd

from evaluation import load_predictions, normalize_label


BENIGN = "Benign"
RANSOMWARE = "Ransomware"

REQUIRED_COLUMNS = [
    "execution_id",
    "predicted_label",
]

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


def validate_predictions(file_path, offline=True, check_argmax=False):
    """
    Validate prediction records before model evaluation.

    Parameters:
        file_path: CSV or JSON prediction file.
        offline: Whether ground-truth labels are required.
        check_argmax: Whether predicted labels must match
                      the highest class probability.

    Returns:
        Validated pandas DataFrame.
    """

    df = load_predictions(file_path)

    if df.empty:
        raise ValueError("Prediction file is empty.")

    # Check required columns.
    required = REQUIRED_COLUMNS.copy()

    if offline:
        required.append("actual_label")

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Validate execution IDs.
    if df["execution_id"].isna().any():
        raise ValueError("Missing execution ID.")

    df = df.copy()

    df["execution_id"] = (
        df["execution_id"].astype(str).str.strip()
    )

    if (df["execution_id"] == "").any():
        raise ValueError("Empty execution ID detected.")

    if df["execution_id"].duplicated().any():
        raise ValueError("Duplicate execution IDs detected.")

    # Validate predicted labels.
    df["predicted_label"] = df["predicted_label"].apply(
        normalize_label
    )

    # Ground-truth labels are mandatory for offline evaluation.
    if offline:
        if df["actual_label"].isna().any():
            raise ValueError(
                "Actual labels are required for offline evaluation."
            )

        df["actual_label"] = df["actual_label"].apply(
            normalize_label
        )

    # Both probability columns must exist together.
    present = [
        column for column in PROBABILITY_COLUMNS
        if column in df.columns
    ]

    if len(present) == 1:
        raise ValueError(
            "Both benign_probability and "
            "ransomware_probability must be provided together."
        )

    # Validate probabilities.
    if len(present) == 2:

        probabilities = df[PROBABILITY_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if probabilities.isna().any().any():
            raise ValueError(
                "Missing or invalid probability values."
            )

        if (
            (probabilities < 0).any().any()
            or (probabilities > 1).any().any()
        ):
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        probability_sums = probabilities.sum(axis=1)

        if not (
            (probability_sums - 1).abs() <= 0.01
        ).all():
            raise ValueError(
                "Class probabilities must sum approximately to 1."
            )

        # Optional baseline argmax consistency check.
        if check_argmax:

            expected = probabilities.idxmax(axis=1).map({
                "benign_probability": BENIGN,
                "ransomware_probability": RANSOMWARE,
            })

            if not (
                expected.values == df["predicted_label"].values
            ).all():
                raise ValueError(
                    "Predicted labels do not match probability argmax."
                )

    # Validate confidence if supplied.
    if "confidence" in df.columns:

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
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

    # Display validation summary.
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
            "predictions.csv [--check-argmax]"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    check_argmax = "--check-argmax" in sys.argv[2:]

    try:
        validate_predictions(
            input_file,
            offline=True,
            check_argmax=check_argmax
        )

    except (ValueError, FileNotFoundError) as error:
        print(f"\nVALIDATION FAILED: {error}")
        sys.exit(1)
