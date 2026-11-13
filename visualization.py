
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

LABELS = ["Benign", "Ransomware"]


def plot_confusion_matrix(results_file):
    """Generate a confusion matrix from evaluation results."""

    with open(results_file, "r", encoding="utf-8") as file:
        results = json.load(file)

    matrix_data = results["confusion_matrix"]

    matrix = [
        [
            matrix_data["true_negative"],
            matrix_data["false_positive"]
        ],
        [
            matrix_data["false_negative"],
            matrix_data["true_positive"]
        ]
    ]

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    vmin=0,
    vmax=2,
    linewidths=1,
    linecolor="white",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cbar=False
    )

    plt.title("Ransomware Detection - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    print("Confusion matrix saved as confusion_matrix.png")


def plot_metric_comparison(results_file):
    """Plot evaluation metrics from one experiment."""

    with open(results_file, "r", encoding="utf-8") as file:
        results = json.load(file)

    metrics = results["metrics"]

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "macro_f1",
        "weighted_f1"
    ]

    values = [metrics[name] for name in metric_names]

    plt.figure(figsize=(9, 5))

    plt.bar(metric_names, values)

    plt.title("Ransomware Detection - Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig("metric_comparison.png", dpi=300)
    plt.show()

    print("Metric chart saved as metric_comparison.png")


if __name__ == "__main__":

    results_file = "evaluation_results.json"

    plot_confusion_matrix(results_file)
    plot_metric_comparison(results_file)
