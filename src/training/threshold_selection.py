"""
Threshold selection utilities for binary classifiers.

The decision threshold must be selected on the validation set only.
The selected threshold can then be frozen and applied to the test set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def collect_binary_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect binary labels and predicted probabilities from a dataset.

    The function supports both single-input and multi-input models.
    """

    true_labels: list[np.ndarray] = []
    predicted_probabilities: list[np.ndarray] = []

    for inputs, labels in dataset:
        probabilities = model(
            inputs,
            training=False,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).reshape(-1)

        labels = np.asarray(
            labels,
            dtype=np.int32,
        ).reshape(-1)

        if len(probabilities) != len(labels):
            raise ValueError(
                "The number of predicted probabilities does not "
                "match the number of labels."
            )

        predicted_probabilities.append(
            probabilities
        )

        true_labels.append(
            labels
        )

    if not true_labels:
        raise ValueError(
            "The dataset did not contain any observations."
        )

    y_true = np.concatenate(
        true_labels
    )

    y_probability = np.concatenate(
        predicted_probabilities
    )

    if not np.all(
        np.isin(
            y_true,
            [0, 1],
        )
    ):
        raise ValueError(
            "Binary labels encoded as 0 and 1 were expected."
        )

    if not np.all(
        np.isfinite(
            y_probability
        )
    ):
        raise ValueError(
            "Predicted probabilities contain non-finite values."
        )

    if np.any(
        (y_probability < 0.0)
        | (y_probability > 1.0)
    ):
        raise ValueError(
            "Predictions must be probabilities between 0 and 1."
        )

    return (
        y_true,
        y_probability,
    )


def threshold_grid_search(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate binary classification metrics across decision thresholds.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    ).reshape(-1)

    y_probability = np.asarray(
        y_probability,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_true) != len(y_probability):
        raise ValueError(
            "Labels and probabilities must have the same length."
        )

    if thresholds is None:
        thresholds = np.arange(
            0.01,
            1.00,
            0.005,
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        y_predicted = (
            y_probability >= threshold
        ).astype(np.int32)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_predicted,
            labels=[0, 1],
        ).ravel()

        sensitivity = recall_score(
            y_true,
            y_predicted,
            zero_division=0,
        )

        specificity = (
            tn / (tn + fp)
            if tn + fp > 0
            else 0.0
        )

        precision = precision_score(
            y_true,
            y_predicted,
            zero_division=0,
        )

        negative_predictive_value = (
            tn / (tn + fn)
            if tn + fn > 0
            else 0.0
        )

        f1 = f1_score(
            y_true,
            y_predicted,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_true,
            y_predicted,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_predicted,
        )

        youden_j = (
            sensitivity
            + specificity
            - 1.0
        )

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
                "precision": float(precision),
                "recall": float(sensitivity),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "negative_predictive_value": float(
                    negative_predictive_value
                ),
                "f1": float(f1),
                "youden_j": float(youden_j),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    return pd.DataFrame(
        rows
    )


def select_threshold_with_minimum_recall(
    results: pd.DataFrame,
    minimum_recall: float = 0.85,
) -> pd.Series:
    """
    Select the threshold with the highest specificity among thresholds
    satisfying a predefined minimum recall.

    F1 and threshold are used as tie-breakers.
    """

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError(
            "minimum_recall must be between 0 and 1."
        )

    eligible_results = results[
        results["recall"] >= minimum_recall
    ].copy()

    if eligible_results.empty:
        maximum_recall = results[
            "recall"
        ].max()

        eligible_results = results[
            results["recall"] == maximum_recall
        ].copy()

        print(
            "Warning: no threshold reached the required "
            f"recall of {minimum_recall:.3f}. "
            f"The maximum observed recall was "
            f"{maximum_recall:.3f}."
        )

    selected_row = (
        eligible_results
        .sort_values(
            by=[
                "specificity",
                "f1",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    return selected_row


def select_threshold_by_youden_j(
    results: pd.DataFrame,
) -> pd.Series:
    """
    Select the threshold maximizing Youden's J statistic.

    Youden J = sensitivity + specificity - 1.
    """

    selected_row = (
        results
        .sort_values(
            by=[
                "youden_j",
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    return selected_row


def plot_threshold_metrics(
    results: pd.DataFrame,
    selected_threshold: float,
    output_path: str | Path,
    title: str = (
        "Validation metrics across decision thresholds"
    ),
) -> None:
    """
    Plot the main validation metrics as functions of the threshold.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results["threshold"],
        results["recall"],
        label="Sensitivity / recall",
    )

    plt.plot(
        results["threshold"],
        results["specificity"],
        label="Specificity",
    )

    plt.plot(
        results["threshold"],
        results["precision"],
        label="Precision",
    )

    plt.plot(
        results["threshold"],
        results["f1"],
        label="F1 score",
    )

    plt.plot(
        results["threshold"],
        results["balanced_accuracy"],
        label="Balanced accuracy",
    )

    plt.axvline(
        x=selected_threshold,
        linestyle="--",
        label=(
            "Selected threshold "
            f"({selected_threshold:.3f})"
        ),
    )

    plt.xlabel(
        "Decision threshold"
    )

    plt.ylabel(
        "Metric value"
    )

    plt.title(
        title
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.ylim(
        0.0,
        1.02,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend(
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_selected_threshold(
    selected_row: pd.Series,
    output_path: str | Path,
    selection_method: str,
    minimum_recall: float | None = None,
) -> None:
    """
    Save the selected threshold and associated validation metrics.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "selection_set": "validation",
        "selection_method": selection_method,
        "selected_threshold": float(
            selected_row["threshold"]
        ),
        "validation_metrics": {
            "accuracy": float(
                selected_row["accuracy"]
            ),
            "balanced_accuracy": float(
                selected_row[
                    "balanced_accuracy"
                ]
            ),
            "precision": float(
                selected_row["precision"]
            ),
            "recall": float(
                selected_row["recall"]
            ),
            "specificity": float(
                selected_row["specificity"]
            ),
            "f1": float(
                selected_row["f1"]
            ),
            "youden_j": float(
                selected_row["youden_j"]
            ),
            "true_negative": int(
                selected_row["true_negative"]
            ),
            "false_positive": int(
                selected_row["false_positive"]
            ),
            "false_negative": int(
                selected_row["false_negative"]
            ),
            "true_positive": int(
                selected_row["true_positive"]
            ),
        },
    }

    if minimum_recall is not None:
        payload["minimum_recall"] = float(
            minimum_recall
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )