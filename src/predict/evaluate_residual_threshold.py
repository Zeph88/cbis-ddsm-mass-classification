"""
Select the residual fusion decision threshold on the validation set.

The test set must not be used in this script.
"""

import gc
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    OUTPUT_MODEL,
    OUTPUT_NPY,
    OUTPUT_PLOT,
    SEED,
)

from src.functions import set_seed, ensure_directory

from src.training.dataset_preparation import (
    train_val_test_sets,
)

from src.training.threshold_selection import (
    collect_binary_predictions,
    plot_threshold_metrics,
    save_selected_threshold,
    select_threshold_by_youden_j,
    select_threshold_with_minimum_recall,
    threshold_grid_search,
)

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

# ======================================================================
# Configuration
# ======================================================================

RESIDUAL_MODEL_PATH = (
    OUTPUT_MODEL
    / f"model_fusion_seed_{SEED}.keras"
)

THRESHOLD_RESULTS_PATH = (
    OUTPUT_MODEL
    / f"residual_threshold_grid_seed_{SEED}.csv"
)

SELECTED_THRESHOLD_PATH = (
    OUTPUT_MODEL
    / f"residual_threshold_seed_{SEED}.json"
)

THRESHOLD_PLOT_PATH = (
    OUTPUT_PLOT
    / f"residual_threshold_metrics_seed_{SEED}.png"
)

MINIMUM_RECALL = 0.85

SELECTION_METHOD = "minimum_recall"

MAMMOGRAM_KEY = [
    "patient_id",
    "left or right breast",
    "image view",
]


# ======================================================================
# Cleanup and reproducibility
# ======================================================================

tf.keras.backend.clear_session()
gc.collect()

set_seed(
    SEED
)


# ======================================================================
# Rebuild the paired dataframe
# ======================================================================

local_height = 384
local_width = 384

global_height = 768
global_width = 512

local_index_path = (
    OUTPUT_NPY
    / (
        f"dataset_index_zoom_"
        f"{local_height}x{local_width}.csv"
    )
)

global_index_path = (
    OUTPUT_NPY
    / (
        f"dataset_index_full_"
        f"{global_height}x{global_width}.csv"
    )
)

local_df = pd.read_csv(
    local_index_path
)

global_df = pd.read_csv(
    global_index_path
)

local_df = local_df.copy()
global_df = global_df.copy()

local_df["local_path"] = local_df[
    "preprocessed_image_path"
]

global_lookup = (
    global_df[
        MAMMOGRAM_KEY
        + ["preprocessed_image_path"]
    ]
    .drop_duplicates(
        subset=MAMMOGRAM_KEY
    )
    .rename(
        columns={
            "preprocessed_image_path": "global_path"
        }
    )
)

paired_df = local_df.merge(
    global_lookup,
    on=MAMMOGRAM_KEY,
    how="inner",
    validate="many_to_one",
)

if paired_df.empty:
    raise ValueError(
        "No local-global pairs were created."
    )


# ======================================================================
# Rebuild the validation dataset
# ======================================================================

train_ds, val_ds, test_ds = train_val_test_sets(
    paired_df,
    path_image="local_path",
    added_path_image="global_path",
    image_height=local_height,
    image_width=local_width,
    added_image_height=global_height,
    added_image_width=global_width,
)

# The threshold selection script must not inspect test observations.
del train_ds
del test_ds

# A small batch reduces memory consumption for the two ResNet50 branches.
val_ds = (
    val_ds
    .unbatch()
    .batch(1)
    .prefetch(1)
)


# ======================================================================
# Load the final selected residual model
# ======================================================================

residual_model = tf.keras.models.load_model(
    RESIDUAL_MODEL_PATH,
    compile=False,
)

print(
    "\nLoaded model:",
    residual_model.name,
)


# ======================================================================
# Validation inference
# ======================================================================

y_val, probability_val = collect_binary_predictions(
    model=residual_model,
    dataset=val_ds,
)

print(
    "\nValidation observations:",
    len(y_val),
)

print(
    "Validation positives:",
    int(y_val.sum()),
)

print(
    "Validation prevalence:",
    float(y_val.mean()),
)

print(
    "Minimum predicted probability:",
    float(probability_val.min()),
)

print(
    "Maximum predicted probability:",
    float(probability_val.max()),
)

print(
    "Average predicted probability:",
    float(probability_val.mean()),
)


# ======================================================================
# Threshold grid search
# ======================================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.005,
)

threshold_results = threshold_grid_search(
    y_true=y_val,
    y_probability=probability_val,
    thresholds=thresholds,
)

threshold_results.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False,
)

print(
    "\nThreshold results saved to:",
    THRESHOLD_RESULTS_PATH,
)


# ======================================================================
# Select one operating point
# ======================================================================

if SELECTION_METHOD == "minimum_recall":
    selected_row = (
        select_threshold_with_minimum_recall(
            results=threshold_results,
            minimum_recall=MINIMUM_RECALL,
        )
    )

elif SELECTION_METHOD == "youden_j":
    selected_row = (
        select_threshold_by_youden_j(
            results=threshold_results,
        )
    )

else:
    raise ValueError(
        "SELECTION_METHOD must be "
        "'minimum_recall' or 'youden_j'."
    )


selected_threshold = float(
    selected_row["threshold"]
)

print(
    "\nSelected validation threshold:",
    selected_threshold,
)

print(
    selected_row[
        [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "youden_j",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    ]
)


# ======================================================================
# Save the selected operating point
# ======================================================================

save_selected_threshold(
    selected_row=selected_row,
    output_path=SELECTED_THRESHOLD_PATH,
    selection_method=SELECTION_METHOD,
    minimum_recall=(
        MINIMUM_RECALL
        if SELECTION_METHOD == "minimum_recall"
        else None
    ),
)

plot_threshold_metrics(
    results=threshold_results,
    selected_threshold=selected_threshold,
    output_path=THRESHOLD_PLOT_PATH,
    title=(
        "Residual fusion validation metrics "
        "across decision thresholds"
    ),
)

print(
    "\nSelected threshold saved to:",
    SELECTED_THRESHOLD_PATH,
)

print(
    "Threshold graph saved to:",
    THRESHOLD_PLOT_PATH,
)


# ======================================================================
# Cleanup
# ======================================================================

del residual_model
del val_ds

tf.keras.backend.clear_session()
gc.collect()