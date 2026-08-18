from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions


def threshold_grid_search(y_true: np.ndarray, y_probability: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:

    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_probability):
        raise ValueError("Labels and probabilities must have the same length.")

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.005)

    thresholds = np.asarray(thresholds, dtype=np.float64)

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_probability, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "negative_predictive_value": metrics["npv"],
                "f1": metrics["f1"],
                "youden_j": metrics["youden_j"],
                "true_negative": metrics["tn"],
                "false_positive": metrics["fp"],
                "false_negative": metrics["fn"],
                "true_positive": metrics["tp"],
            }
        )

    return pd.DataFrame(
        rows
    )


def select_threshold_with_minimum_recall(results: pd.DataFrame, minimum_recall: float = 0.85) -> pd.Series:

    eligible_results = results[results["recall"] >= minimum_recall].copy()

    if eligible_results.empty:

        maximum_recall = results["recall"].max()
        eligible_results = results[results["recall"] == maximum_recall].copy()

        print(f"Warning: no threshold reached the required recall of {minimum_recall:.3f}. The maximum observed recall was {maximum_recall:.3f}.")

    selected_row = (eligible_results.sort_values(by=["specificity", "f1", "threshold"], ascending=[False, False, False]).iloc[0])

    return selected_row


def select_threshold_by_youden_j(results: pd.DataFrame) -> pd.Series:

    selected_row = (results.sort_values(by=["youden_j", "balanced_accuracy", "threshold"], ascending=[False, False, False]).iloc[0])

    return selected_row


def plot_threshold_metrics(results, selected_threshold, output_path, title = ("Validation metrics across decision thresholds")):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(results["threshold"], results["recall"], label="Sensitivity / recall")
    plt.plot(results["threshold"], results["specificity"], label="Specificity")
    plt.plot(results["threshold"], results["precision"], label="Precision")
    plt.plot(results["threshold"], results["f1"], label="F1 score")
    plt.plot(results["threshold"], results["balanced_accuracy"], label="Balanced accuracy")
    plt.axvline(x=selected_threshold, linestyle="--", label=(f"Selected threshold ({selected_threshold:.3f})"))
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def save_selected_threshold(selected_row: pd.Series, output_path: str | Path, selection_method: str, minimum_recall: float | None = None):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "selection_set": "validation",
        "selection_method": selection_method,
        "selected_threshold": float(selected_row["threshold"]),
        "validation_metrics": {
            "accuracy": float(selected_row["accuracy"]),
            "balanced_accuracy": float(selected_row["balanced_accuracy"]),
            "precision": float(selected_row["precision"]),
            "recall": float(selected_row["recall"]),
            "specificity": float(selected_row["specificity"]),
            "f1": float(selected_row["f1"]),
            "youden_j": float(selected_row["youden_j"]),
            "true_negative": int(selected_row["true_negative"]),
            "false_positive": int(selected_row["false_positive"]),
            "false_negative": int(selected_row["false_negative"]),
            "true_positive": int(selected_row["true_positive"])
        }
    }

    if minimum_recall is not None:
        payload["minimum_recall"] = float(minimum_recall)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)