import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from src.config import OUTPUT_MODEL, SEED, N_BOOTSTRAP, N_CALIBRATION_BINS, ARCHITECTURES, N_OUTER_FOLDS, THRESHOLDS
from src.functions import ensure_directory


# Configuration
CV_ROOT = (OUTPUT_MODEL / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{SEED}")

FOLDS_DIR = CV_ROOT / "folds"
RESULTS_DIR = CV_ROOT / "results"

ensure_directory(RESULTS_DIR)

# Metrics
def calculate_metrics(y_true, probability, threshold=None):

    metrics = {
        "auc": roc_auc_score(y_true, probability),
        "bce": log_loss(y_true, probability, labels=[0, 1]),
        "ap": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }

    if threshold is None:
        return metrics

    y_pred = (probability >= threshold).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (recall + specificity) / 2
    npv = tn / (tn + fn) if tn + fn else 0.0
    youden_j = recall + specificity - 1

    metrics.update({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "npv" : npv,
        "youden_j" : youden_j,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    })

    return metrics

# Collect binary labels and predicted probabilities from a dataset.
def collect_binary_predictions(model: tf.keras.Model, dataset: tf.data.Dataset, input_selector=None) -> tuple[np.ndarray, np.ndarray]:

    true_labels: list[np.ndarray] = []
    predicted_probabilities: list[np.ndarray] = []

    for inputs, labels in dataset:

        if input_selector is not None:
            inputs = input_selector(inputs)
            
        probabilities = model(inputs, training=False)
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        if len(probabilities) != len(labels):
            raise ValueError("The number of predicted probabilities does not match the number of labels.")

        predicted_probabilities.append(probabilities)
        true_labels.append(labels)

    y_true = np.concatenate(true_labels)
    y_probability = np.concatenate(predicted_probabilities)

    return y_true, y_probability

def build_binary_metrics(thresholds=THRESHOLDS):
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]

    for threshold in thresholds:
        suffix = int(threshold * 100)

        metrics.extend([
            tf.keras.metrics.Recall(
                name=f"recall_{suffix}",
                thresholds=threshold,
            ),
            tf.keras.metrics.Precision(
                name=f"precision_{suffix}",
                thresholds=threshold,
            ),
        ])

    return metrics

def plot_training_metric(history, metric_name, ylabel, title, output_path):

    train_key = metric_name
    validation_key = f"val_{metric_name}"

    if train_key not in history.history:
        raise KeyError(f"Metric '{train_key}' was not found in history. Available metrics: {list(history.history.keys())}")

    if validation_key not in history.history:
        raise KeyError(f"Metric '{validation_key}' was not found in history. Available metrics: {list(history.history.keys())}")

    train_values = history.history[train_key]
    validation_values = history.history[validation_key]

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_values, label=f"Training {metric_name}")
    plt.plot(epochs, validation_values, label=f"Validation {metric_name}")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    print(f"Graph saved to: {output_path}")


def validate_model_path(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

def load_preprocessed_indexes(dev_patient_ids=None):
    local_path = OUTPUT_NPY / f"dataset_index_zoom_{LOCAL_HEIGHT}x{LOCAL_WIDTH}.csv"
    global_path = OUTPUT_NPY / f"dataset_index_full_{GLOBAL_HEIGHT}x{GLOBAL_WIDTH}.csv"
    local_df, global_df = load_data(local_path, global_path)

    local_df["patient_id"] = local_df["patient_id"].astype(str)
    global_df["patient_id"] = global_df["patient_id"].astype(str)

    if "label" in local_df.columns:
        local_df["label"] = local_df["label"].astype(int)

    if dev_patient_ids is not None:
        local_df = local_df[local_df["patient_id"].isin(dev_patient_ids)].copy()
        global_df = global_df[global_df["patient_id"].isin(dev_patient_ids)].copy()

    return local_df, global_df