import os
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import OUTPUT_NPY, OUTPUT_MODEL, OUTPUT_PLOT, SEED, MAMMOGRAM_KEY, EPOCHS, THRESHOLDS

from pathlib import Path
import matplotlib.pyplot as plt

from src.functions import set_seed, ensure_directory, load_data, parse_arguments
from src.training.dataset_preparation import train_val_test_sets
from src.modeling.fusion import build_residual_fusion, build_symmetric_fusion
from src.evaluation.evaluation_utils import plot_training_metric, build_binary_metrics


ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

args = parse_arguments(
    description=(
        "Train a local/global fusion model."
    ),
    arguments=[
        {
            "name": "--model",
            "choices": [
                "residual",
                "symmetric",
            ],
            "required": True,
            "help": (
                "Fusion architecture to train: "
                "'residual' or 'symmetric'."
            ),
        },
    ],
)

LOCAL_MODEL_PATH = OUTPUT_MODEL / "local_resnet50_head.keras"
GLOBAL_MODEL_PATH = OUTPUT_MODEL / "global_resnet50_head.keras"
FUSION_MODEL_PATH = OUTPUT_MODEL / f"model_fusion_{args.model}_seed_{SEED}.keras"
FUSION_LOG_PATH = OUTPUT_MODEL / f"model_fusion_{args.model}_seed_{SEED}.csv"

LOCAL_EMBEDDING_LAYER = "mammography_adapter"
GLOBAL_EMBEDDING_LAYER = "mammography_adapter_relu"

FUSION_UNITS = 16
FUSION_DROPOUT = 0.5
FUSION_L2 = 1e-4

RESIDUAL_UNITS = 8
RESIDUAL_DROPOUT = 0.3
RESIDUAL_L2 = 1e-4

FUSION_LEARNING_RATE = 1e-4

tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)

# Load trained local and global branches
print("\nLoading trained branches...")

local_model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
global_model = tf.keras.models.load_model(GLOBAL_MODEL_PATH, compile=False)

print("Local input shape:", local_model.input_shape)
print("Global input shape:", global_model.input_shape)

local_height = local_model.input_shape[1]
local_width = local_model.input_shape[2]

global_height = global_model.input_shape[1]
global_width = global_model.input_shape[2]

local_index_path = OUTPUT_NPY / f"dataset_index_zoom_{local_height}x{local_width}.csv"
global_index_path = OUTPUT_NPY / f"dataset_index_full_{global_height}x{global_width}.csv"

print("\nLocal index:", local_index_path)
print("Global index:", global_index_path)

local_df, global_df = load_data(local_index_path, global_index_path)

# Pair every local lesion with its full mammogram
required_local_columns = MAMMOGRAM_KEY + ["preprocessed_image_path", "label"]
required_global_columns = MAMMOGRAM_KEY + ["preprocessed_image_path"]

missing_local_columns = [column for column in required_local_columns if column not in local_df.columns]
missing_global_columns = [column for column in required_global_columns if column not in global_df.columns]

if missing_local_columns:
    raise ValueError(f"Missing local columns: {missing_local_columns}")

if missing_global_columns:
    raise ValueError(f"Missing global columns: {missing_global_columns}")

local_df = local_df.copy()
global_df = global_df.copy()

local_df["local_path"] = local_df["preprocessed_image_path"]

# Confirm that each mammogram key refers to only one full-image path.
global_path_count = global_df.groupby(MAMMOGRAM_KEY)["preprocessed_image_path"].nunique()
conflicting_global_paths = global_path_count[global_path_count > 1]

if not conflicting_global_paths.empty:
    raise ValueError(f"Some mammogram keys refer to several different global image paths: {conflicting_global_paths.head()}")

# This works whether the global index contains one row per mammogram or repeated rows originating from a lesion-level index.
global_lookup = global_df[MAMMOGRAM_KEY + ["preprocessed_image_path"]].drop_duplicates(subset=MAMMOGRAM_KEY).rename(columns={"preprocessed_image_path": "global_path"})

initial_local_count = len(local_df)

df = local_df.merge(global_lookup, on=MAMMOGRAM_KEY, how="inner", validate="many_to_one")

print("\nInitial local lesion count:", initial_local_count)

print("Successfully paired lesion count:", len(df))

print("Unmatched local lesions:", initial_local_count - len(df))

if df.empty:
    raise ValueError("No local lesions could be paired with a global image.")

print("\nPaired dataframe preview:")

print(df[MAMMOGRAM_KEY + ["local_path", "global_path", "label"]].head())


# Build paired TensorFlow datasets
train_ds, val_ds, test_ds = train_val_test_sets(
    df,
    path_image="local_path",
    added_path_image="global_path",
    image_height=local_height,
    image_width=local_width,
    added_image_height=global_height,
    added_image_width=global_width,
)


print("\nDataset element specification:")

print(train_ds.element_spec)


# Validate one paired batch.
for images, labels in train_ds.take(1):

    local_images, global_images = images

    print("\nLocal batch shape:", local_images.shape)
    print("Global batch shape:", global_images.shape)
    print("Label batch shape:", labels.shape)
    print("Expected local model shape:", local_model.input_shape)
    print("Expected global model shape:", global_model.input_shape)

    if (tuple(local_images.shape[1:]) != tuple(local_model.input_shape[1:])):
        raise ValueError("The local dataset shape does not match the local model input shape.")

    if (tuple(global_images.shape[1:]) != tuple(global_model.input_shape[1:])):
        raise ValueError("The global dataset shape does not match the global model input shape.")


# Build fusion model
if args.model == "residual":

    fusion_model = build_residual_fusion(
        local_model=local_model,
        global_model=global_model,
        correction_units=RESIDUAL_UNITS,
        correction_dropout=RESIDUAL_DROPOUT,
        correction_l2=RESIDUAL_L2,
    )

elif args.model == "symmetric":

    fusion_model = build_symmetric_fusion(
        local_model=local_model,
        global_model=global_model,
        fusion_units=FUSION_UNITS,
        fusion_dropout=FUSION_DROPOUT,
        fusion_l2=FUSION_L2,
    )

fusion_model.summary()

# Verify that the initial residual model matches the local model

# Extract one individual local-global pair instead of one full batch.

if args.model == "residual":
    
    verification_ds = (val_ds.unbatch().take(16).batch(16))

    for images, labels in verification_ds:

        local_images, global_images = images

        print("Verification local shape:", local_images.shape)
        print("Verification global shape:", global_images.shape)

        local_probabilities = local_model(local_images, training=False).numpy().reshape(-1)
        residual_probabilities = fusion_model([local_images, global_images], training=False).numpy().reshape(-1)
        maximum_difference = np.max(np.abs(local_probabilities - residual_probabilities))

        print("Local initial probability:", local_probabilities)
        print("Residual initial probability:", residual_probabilities)
        print("Maximum initial local/residual probability difference:", maximum_difference)

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
    metrics=build_binary_metrics()
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

# Plot training history
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
        / f"fusion_{args.model}_loss_seed_{SEED}.png"
    )
)


plot_training_metric(
    history=history,
    metric_name="auc",
    ylabel="ROC AUC",
    title="Fusion model training and validation ROC AUC",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_{args.model}_auc_seed_{SEED}.png"
    ),
)


plot_training_metric(
    history=history,
    metric_name="accuracy",
    ylabel="Accuracy",
    title="Fusion model training and validation accuracy",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_{args.model}_accuracy_seed_{SEED}.png"
    ),
)

plot_training_metric(
    history=history,
    metric_name="pr_auc",
    ylabel="PR AUC",
    title="Fusion model training and validation PR AUC",
    output_path=(
        OUTPUT_PLOT
        / f"fusion_{args.model}_pr_auc_seed_{SEED}.png"
    ),
)


# Final cleanup
del history
del fusion_model
del local_model
del global_model

del train_ds
del val_ds
del test_ds

tf.keras.backend.clear_session()
gc.collect()