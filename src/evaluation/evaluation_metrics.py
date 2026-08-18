import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from src.config import OUTPUT_MODEL, SEED, N_BOOTSTRAP, N_CALIBRATION_BINS, ARCHITECTURES, N_OUTER_FOLDS
from src.functions import ensure_directory


# Configuration
CV_ROOT = (OUTPUT_MODEL / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{SEED}")

FOLDS_DIR = CV_ROOT / "folds"
RESULTS_DIR = CV_ROOT / "results"

ensure_directory(RESULTS_DIR)

# Metrics
def calculate_metrics(y_true, probability):
    return {
        "auc": roc_auc_score(y_true, probability),
        "bce": log_loss(y_true, probability, labels=[0, 1]),
        "ap": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }