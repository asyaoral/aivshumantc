"""Held-out evaluation and result export."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_holdout(model, selected: pd.Series, dataset_name: str, file_stem: str,
                     features: pd.DataFrame, labels: pd.Series,
                     output_dir: Path) -> tuple[dict, pd.DataFrame]:
    predictions = model.predict(features)
    metrics = {
        "Dataset": dataset_name,
        "Selected experiment": selected["Experiment"],
        "Train CV F1 (macro)": selected["CV F1 (macro)"],
        "Accuracy": accuracy_score(labels, predictions),
        "Precision (macro)": precision_score(labels, predictions, average="macro", zero_division=0),
        "Recall (macro)": recall_score(labels, predictions, average="macro", zero_division=0),
        "F1 (macro)": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    print(f"\n{dataset_name.upper()}")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["AI", "Human"],
        zero_division=0,
    ))

    results = features.copy()
    results["Actual"] = labels.values
    results["Predicted"] = predictions
    results["Correct"] = results["Actual"] == results["Predicted"]
    errors = results[~results["Correct"]].copy()
    errors["Actual name"] = errors["Actual"].map({0: "AI", 1: "Human"})
    errors["Predicted name"] = errors["Predicted"].map({0: "AI", 1: "Human"})
    results.to_csv(output_dir / f"{file_stem}_predictions.csv", index=False)
    errors.to_csv(output_dir / f"{file_stem}_misclassified_texts.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=[0, 1],
        display_labels=["AI", "Human"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"{dataset_name} confusion matrix — selected model")
    plt.tight_layout()
    plt.savefig(output_dir / f"{file_stem}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics, errors

