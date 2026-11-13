
import tempfile
from pathlib import Path

import pandas as pd

from validate_predictions import validate_predictions


def make_valid_data():
    """Create two valid execution-level prediction records."""

    return pd.DataFrame({
        "execution_id": ["EXEC001", "EXEC002"],
        "actual_label": ["Benign", "Ransomware"],
        "predicted_label": ["Benign", "Ransomware"],
        "benign_probability": [0.95, 0.10],
        "ransomware_probability": [0.05, 0.90],
        "confidence": [0.95, 0.90]
    })


def run_test(test_name, data, should_fail=False):
    """Save test data to CSV and pass its file path to the validator."""

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = Path(temp_dir) / "test_predictions.csv"

            # Create a temporary CSV file.
            data.to_csv(file_path, index=False)

            # IMPORTANT: Pass the file path, NOT the DataFrame.
            validate_predictions(
                str(file_path),
                offline=True
            )

        if should_fail:
            print(f"FAILED: {test_name}")
            print("Invalid data was incorrectly accepted.")

        else:
            print(f"PASSED: {test_name}")

    except ValueError as error:

        if should_fail:
            print(f"PASSED: {test_name}")
            print(f"  Correctly rejected: {error}")

        else:
            print(f"FAILED: {test_name}")
            print(f"  Unexpected error: {error}")

    except Exception as error:
        print(f"ERROR: {test_name}")
        print(f"  Unexpected problem: {error}")


def main():

    print("\n===== PREDICTION VALIDATION TESTS =====\n")

    # TEST 1: Valid predictions should be accepted.
    run_test(
        "Valid predictions",
        make_valid_data()
    )

    # TEST 2: Duplicate execution IDs should be rejected.
    duplicate = make_valid_data()
    duplicate.loc[1, "execution_id"] = "EXEC001"

    run_test(
        "Duplicate execution IDs",
        duplicate,
        should_fail=True
    )

    # TEST 3: Invalid class labels should be rejected.
    invalid_label = make_valid_data()
    invalid_label.loc[0, "predicted_label"] = "Unknown"

    run_test(
        "Invalid class label",
        invalid_label,
        should_fail=True
    )

    # TEST 4: Invalid probability sums should be rejected.
    invalid_probability = make_valid_data()
    invalid_probability.loc[0, "benign_probability"] = 0.70

    run_test(
        "Invalid probability sum",
        invalid_probability,
        should_fail=True
    )

    # TEST 5: Missing actual labels should be rejected offline.
    missing_label = make_valid_data()
    missing_label.loc[0, "actual_label"] = None

    run_test(
        "Missing actual label",
        missing_label,
        should_fail=True
    )

    print("\n===== TESTING COMPLETED =====")


if __name__ == "__main__":
    main()
