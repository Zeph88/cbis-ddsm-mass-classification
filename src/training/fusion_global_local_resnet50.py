import os
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

from src.config import OUTPUT_NPY, OUTPUT_MODEL, OUTPUT_PLOT, SEED, MAMMOGRAM_KEY

from pathlib import Path
import matplotlib.pyplot as plt

from src.functions import set_seed, ensure_directory, load_data

from src.training.dataset_preparation import (
    train_val_test_sets,
)

from src.modeling.fusion import (
    build_residual_fusion,
    build_symmetric_fusion,
)

# ======================================================================
# Configuration
# ======================================================================

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

LOCAL_MODEL_PATH = (
    OUTPUT_MODEL / "local_resnet50_head.keras"
)

GLOBAL_MODEL_PATH = (
    OUTPUT_MODEL / "global_resnet50_head.keras"
)

FUSION_MODEL_PATH = (
    OUTPUT_MODEL / f"model_fusion_seed_{SEED}.keras"
)

FUSION_LOG_PATH = (
    OUTPUT_MODEL / f"model_fusion_seed_{SEED}.csv"
)

LOCAL_EMBEDDING_LAYER = "mammography_adapter"
GLOBAL_EMBEDDING_LAYER = "mammography_adapter_relu"

FUSION_UNITS = 16
FUSION_DROPOUT = 0.5
FUSION_L2 = 1e-4
FUSION_LEARNING_RATE = 1e-4

EPOCHS = 100

THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


# ======================================================================
# Initial cleanup and reproducibility
# ======================================================================

tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)



def plot_training_metric(
    history,
    metric_name,
    ylabel,
    title,
    output_path,
):
    """
    Plot and save the training and validation values of one metric.
    """

    train_key = metric_name
    validation_key = f"val_{metric_name}"

    if train_key not in history.history:
        raise KeyError(
            f"Metric '{train_key}' was not found in history. "
            f"Available metrics: {list(history.history.keys())}"
        )

    if validation_key not in history.history:
        raise KeyError(
            f"Metric '{validation_key}' was not found in history. "
            f"Available metrics: {list(history.history.keys())}"
        )

    train_values = history.history[train_key]
    validation_values = history.history[validation_key]

    epochs = range(
        1,
        len(train_values) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_values,
        label=f"Training {ylabel}",
    )

    plt.plot(
        epochs,
        validation_values,
        label=f"Validation {ylabel}",
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(
        f"Graph saved to: {output_path}"
    )

# ======================================================================
# Load trained local and global branches
# ======================================================================

print("\nLoading trained branches...")

local_model = tf.keras.models.load_model(
    LOCAL_MODEL_PATH,
    compile=False,
)

global_model = tf.keras.models.load_model(
    GLOBAL_MODEL_PATH,
    compile=False,
)

print(
    "Local input shape:",
    local_model.input_shape,
)

print(
    "Global input shape:",
    global_model.input_shape,
)

local_height = local_model.input_shape[1]
local_width = local_model.input_shape[2]

global_height = global_model.input_shape[1]
global_width = global_model.input_shape[2]

local_index_path = OUTPUT_NPY / f"dataset_index_zoom_{local_height}x{local_width}.csv"
global_index_path = OUTPUT_NPY / f"dataset_index_full_{global_height}x{global_width}.csv"

print(
    "\nLocal index:",
    local_index_path,
)

print(
    "Global index:",
    global_index_path,
)

local_df, global_df = load_data(local_index_path, global_index_path)

# ======================================================================
# Pair every local lesion with its full mammogram
# ======================================================================

required_local_columns = (
    MAMMOGRAM_KEY
    + [
        "preprocessed_image_path",
        "label",
    ]
)

required_global_columns = (
    MAMMOGRAM_KEY
    + [
        "preprocessed_image_path",
    ]
)

missing_local_columns = [
    column
    for column in required_local_columns
    if column not in local_df.columns
]

missing_global_columns = [
    column
    for column in required_global_columns
    if column not in global_df.columns
]

if missing_local_columns:
    raise ValueError(
        "Missing local columns: "
        f"{missing_local_columns}"
    )

if missing_global_columns:
    raise ValueError(
        "Missing global columns: "
        f"{missing_global_columns}"
    )


local_df = local_df.copy()
global_df = global_df.copy()

local_df["local_path"] = local_df[
    "preprocessed_image_path"
]


# Confirm that each mammogram key refers to only one full-image path.
global_path_count = (
    global_df
    .groupby(MAMMOGRAM_KEY)[
        "preprocessed_image_path"
    ]
    .nunique()
)

conflicting_global_paths = global_path_count[
    global_path_count > 1
]

if not conflicting_global_paths.empty:
    raise ValueError(
        "Some mammogram keys refer to several different "
        "global image paths:\n"
        f"{conflicting_global_paths.head()}"
    )


# This works whether the global index contains one row per mammogram
# or repeated rows originating from a lesion-level index.
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


initial_local_count = len(
    local_df
)

df = local_df.merge(
    global_lookup,
    on=MAMMOGRAM_KEY,
    how="inner",
    validate="many_to_one",
)

print(
    "\nInitial local lesion count:",
    initial_local_count,
)

print(
    "Successfully paired lesion count:",
    len(df),
)

print(
    "Unmatched local lesions:",
    initial_local_count - len(df),
)

if df.empty:
    raise ValueError(
        "No local lesions could be paired with a global image."
    )

print(
    "\nPaired dataframe preview:"
)

print(
    df[
        MAMMOGRAM_KEY
        + [
            "local_path",
            "global_path",
            "label",
        ]
    ].head()
)


# ======================================================================
# Build paired TensorFlow datasets
# ======================================================================

train_ds, val_ds, test_ds = train_val_test_sets(
    df,
    path_image="local_path",
    added_path_image="global_path",
    image_height=local_height,
    image_width=local_width,
    added_image_height=global_height,
    added_image_width=global_width,
)


print(
    "\nDataset element specification:"
)

print(
    train_ds.element_spec
)


# Validate one paired batch.
for images, labels in train_ds.take(1):
    local_images, global_images = images

    print(
        "\nLocal batch shape:",
        local_images.shape,
    )

    print(
        "Global batch shape:",
        global_images.shape,
    )

    print(
        "Label batch shape:",
        labels.shape,
    )

    print(
        "Expected local model shape:",
        local_model.input_shape,
    )

    print(
        "Expected global model shape:",
        global_model.input_shape,
    )

    if (
        tuple(local_images.shape[1:])
        != tuple(local_model.input_shape[1:])
    ):
        raise ValueError(
            "The local dataset shape does not match "
            "the local model input shape."
        )

    if (
        tuple(global_images.shape[1:])
        != tuple(global_model.input_shape[1:])
    ):
        raise ValueError(
            "The global dataset shape does not match "
            "the global model input shape."
        )


# ======================================================================
# Build fusion model
# ======================================================================

# fusion_model = build_symmetric_fusion(
#     local_model=local_model,
#     global_model=global_model,
#     fusion_units=16,
#     fusion_dropout=0.5,
#     fusion_l2=1e-4,
# )

fusion_model = build_residual_fusion(
    local_model=local_model,
    global_model=global_model,
    correction_units=8,
    correction_dropout=0.3,
    correction_l2=1e-4,
)

fusion_model.summary()

# ======================================================================
# Verify that the initial residual model matches the local model
# ======================================================================

# Extract one individual local-global pair instead of one full batch.
verification_ds = (
    val_ds
    .unbatch()
    .take(1)
    .batch(1)
)

for images, labels in verification_ds:
    local_images, global_images = images

    print(
        "Verification local shape:",
        local_images.shape,
    )

    print(
        "Verification global shape:",
        global_images.shape,
    )

    local_probabilities = local_model(
        local_images,
        training=False,
    ).numpy().reshape(-1)

    residual_probabilities = fusion_model(
        [
            local_images,
            global_images,
        ],
        training=False,
    ).numpy().reshape(-1)

    maximum_difference = np.max(
        np.abs(
            local_probabilities
            - residual_probabilities
        )
    )

    print(
        "Local initial probability:",
        local_probabilities,
    )

    print(
        "Residual initial probability:",
        residual_probabilities,
    )

    print(
        "Maximum initial local/residual probability difference:",
        maximum_difference,
    )

    if maximum_difference > 1e-5:
        raise ValueError(
            "The residual model does not initially reproduce "
            "the local model."
        )

del verification_ds
gc.collect()

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FUSION_LEARNING_RATE,
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
        ),
        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC",
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
        tf.keras.metrics.Recall(
            name="recall_40",
            thresholds=0.40,
        ),
        tf.keras.metrics.Precision(
            name="precision_40",
            thresholds=0.40,
        ),
        tf.keras.metrics.Recall(
            name="recall_45",
            thresholds=0.45,
        ),
        tf.keras.metrics.Precision(
            name="precision_45",
            thresholds=0.45,
        ),
        tf.keras.metrics.Recall(
            name="recall_50",
            thresholds=0.50,
        ),
        tf.keras.metrics.Precision(
            name="precision_50",
            thresholds=0.50,
        ),
        tf.keras.metrics.Recall(
            name="recall_55",
            thresholds=0.55,
        ),
        tf.keras.metrics.Precision(
            name="precision_55",
            thresholds=0.55,
        ),
    ],
)

fusion_model.summary()


print(
    "\nTrainable parameters:",
    int(
        np.sum(
            [
                np.prod(variable.shape)
                for variable
                in fusion_model.trainable_weights
            ]
        )
    ),
)


# ======================================================================
# Train the fusion head
# ======================================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        FUSION_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
    ),

    tf.keras.callbacks.CSVLogger(
        FUSION_LOG_PATH
    ),
]

history = fusion_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ======================================================================
# Plot training history
# ======================================================================

Path(OUTPUT_PLOT).mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "\nAvailable history metrics:"
)

print(
    list(history.history.keys())
)


plot_training_metric(
    history=history,
    metric_name="loss",
    ylabel="Loss",
    title="Fusion model training and validation loss",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_loss_seed_{SEED}.png"
    ),
)


plot_training_metric(
    history=history,
    metric_name="auc",
    ylabel="ROC AUC",
    title="Fusion model training and validation ROC AUC",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_auc_seed_{SEED}.png"
    ),
)


plot_training_metric(
    history=history,
    metric_name="accuracy",
    ylabel="Accuracy",
    title="Fusion model training and validation accuracy",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_accuracy_seed_{SEED}.png"
    ),
)

plot_training_metric(
    history=history,
    metric_name="pr_auc",
    ylabel="PR AUC",
    title="Fusion model training and validation PR AUC",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_pr_auc_seed_{SEED}.png"
    ),
)


# ======================================================================
# Final cleanup
# ======================================================================

del history
del fusion_model
del local_model
del global_model

del train_ds
del val_ds
del test_ds

tf.keras.backend.clear_session()
gc.collect()