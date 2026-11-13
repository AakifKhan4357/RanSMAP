
import json
from pathlib import Path

from evaluation import load_predictions, evaluate_predictions


def calculate_retention(before_file, after_file):
    """
    Compare model performance on the SAME fixed test
    executions before and after adaptation.

    Both files must contain predictions for the same
    original-domain held-out executions.
    """

    before = load_predictions(before_file)
    after = load_predictions(after_file)

    # Check that both evaluations use the same executions.
    if before["execution_id"].isna().any() or after["execution_id"].isna().any():
        raise ValueError("Missing execution IDs detected.")

    if before["execution_id"].duplicated().any():
        raise ValueError("Duplicate execution IDs in before-adaptation data.")

    if after["execution_id"].duplicated().any():
        raise ValueError("Duplicate execution IDs in after-adaptation data.")

    before = before.set_index("execution_id").sort_index()
    after = after.set_index("execution_id").sort_index()

    if not before.index.equals(after.index):
        raise ValueError(
            "Before and after files must contain exactly "
            "the same execution IDs."
        )

    # Ground truth must remain identical.
    if before["actual_label"].isna().any() or after["actual_label"].isna().any():
        raise ValueError("Missing ground-truth labels detected.")

    if not before["actual_label"].astype(str).str.strip().str.lower().equals(
        after["actual_label"].astype(str).str.strip().str.lower()
    ):
        raise ValueError("Ground-truth labels do not match.")

    # Evaluate both model versions.
    before_results = evaluate_predictions(before.reset_index())
    after_results = evaluate_predictions(after.reset_index())

    before_f1 = before_results["metrics"]["f1_score"]
    after_f1 = after_results["metrics"]["f1_score"]

    retention_change = after_f1 - before_f1
    forgetting = before_f1 - after_f1

    results = {
        "evaluation_domain": "original_domain",
        "evaluation_unit": "execution",
        "sample_count": before_results["sample_count"],
        "f1_before_adaptation": before_f1,
        "f1_after_adaptation": after_f1,
        "retention_change": retention_change,
        "forgetting": forgetting
    }

    return results


def main():

    before_file = "before_adaptation.csv"
    after_file = "after_adaptation.csv"

    if not Path(before_file).exists() or not Path(after_file).exists():
        print("Prediction files are not available yet.")
        print("Expected files:")
        print(" - before_adaptation.csv")
        print(" - after_adaptation.csv")
        return

    results = calculate_retention(before_file, after_file)

    with open(
        "retention_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n===== ADAPTATION RETENTION RESULTS =====")
    print(json.dumps(results, indent=4))
    print("\nResults saved to retention_results.json")


if __name__ == "__main__":
    main()
