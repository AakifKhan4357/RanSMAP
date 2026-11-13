
import pandas as pd

from evaluation import load_predictions, evaluate_predictions


def validate_same_test_executions(prediction_data):
    """
    Ensure all experiments use exactly the same
    held-out execution IDs and ground-truth labels.
    """

    reference_name = None
    reference_labels = None

    for experiment_name, df in prediction_data.items():

        if df["execution_id"].isna().any():
            raise ValueError(
                f"Missing execution ID in {experiment_name}"
            )

        if df["execution_id"].duplicated().any():
            raise ValueError(
                f"Duplicate execution IDs in {experiment_name}"
            )

        if df["actual_label"].isna().any():
            raise ValueError(
                f"Missing actual labels in {experiment_name}"
            )

        current_labels = {
            str(row["execution_id"]): str(row["actual_label"]).strip().lower()
            for _, row in df.iterrows()
        }

        if reference_name is None:
            reference_name = experiment_name
            reference_labels = current_labels
            continue

        if set(current_labels) != set(reference_labels):
            raise ValueError(
                f"{experiment_name} does not use the same "
                f"test executions as {reference_name}."
            )

        for execution_id, label in current_labels.items():
            if label != reference_labels[execution_id]:
                raise ValueError(
                    f"Ground-truth mismatch for execution "
                    f"{execution_id}."
                )

    print("Validation passed: all experiments use the same test executions.")


def compare_models(prediction_files):
    """
    Evaluate and compare multiple model experiments.
    """

    prediction_data = {}

    # Load all prediction files first.
    for experiment_name, file_path in prediction_files.items():

        print(f"Loading: {experiment_name}")

        prediction_data[experiment_name] = load_predictions(file_path)

    # Validate the test executions before comparing metrics.
    validate_same_test_executions(prediction_data)

    comparison_results = []

    for experiment_name, predictions in prediction_data.items():

        print(f"Evaluating: {experiment_name}")

        results = evaluate_predictions(predictions)
        metrics = results["metrics"]

        comparison_results.append({
            "experiment": experiment_name,
            "sample_count": results["sample_count"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
        })

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df.to_csv(
        "model_comparison.csv",
        index=False
    )

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df.to_string(index=False))

    print("\nComparison saved to model_comparison.csv")

    return comparison_df


if __name__ == "__main__":

    prediction_files = {
        "Static Baseline": "static_predictions.csv",
        "Naive Fine-Tuning": "finetuning_predictions.csv",
        "Replay Adaptation": "replay_predictions.csv",
    }

    compare_models(prediction_files)
