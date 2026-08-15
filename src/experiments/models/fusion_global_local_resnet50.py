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

from src.config import (
    OUTPUT_NPY,
    OUTPUT_MODEL,
    OUTPUT_PLOT,
    SEED,
)

from pathlib import Path
import matplotlib.pyplot as plt

from src.functions import set_seed, ensure_directory

from src.training.dataset_preparation import (
    train_val_test_sets,
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

MAMMOGRAM_KEY = [
    "patient_id",
    "left or right breast",
    "image view",
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
# Fusion model
# ======================================================================

def fusion_models(
    local_model,
    global_model,
    local_embedding_layer_name,
    global_embedding_layer_name,
):
    """
    Build a lesion-level fusion model from frozen local and global
    ResNet50 branches.

    The local branch provides lesion morphology.
    The global branch provides mammographic context.
    """

    regularizer = tf.keras.regularizers.l2(
        FUSION_L2
    )

    # --------------------------------------------------------------
    # Extract the trained embeddings
    # --------------------------------------------------------------

    try:
        local_embedding_layer = local_model.get_layer(
            local_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local embedding layer "
            f"'{local_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    try:
        global_embedding_layer = global_model.get_layer(
            global_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Global embedding layer "
            f"'{global_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in global_model.layers]}"
        ) from exc

    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=local_embedding_layer.output,
        name="local_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_embedding_layer.output,
        name="global_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    print(
        "Local embedding shape:",
        local_extractor.output_shape,
    )

    print(
        "Global embedding shape:",
        global_extractor.output_shape,
    )

    if len(local_extractor.output_shape) != 2:
        raise ValueError(
            "The local extractor must return a flat embedding. "
            f"Received {local_extractor.output_shape}."
        )

    if len(global_extractor.output_shape) != 2:
        raise ValueError(
            "The global extractor must return a flat embedding. "
            f"Received {global_extractor.output_shape}."
        )

    # --------------------------------------------------------------
    # Define fusion inputs
    # --------------------------------------------------------------

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    # training=False disables branch augmentation and dropout,
    # and keeps their BatchNormalization layers in inference mode.
    local_embedding = local_extractor(
        local_input,
        training=False,
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    # Normalize each representation independently so that one branch
    # does not dominate merely because of a different feature scale.
    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    # Expected dimensions:
    # local embedding  = 16
    # global embedding = 8
    # concatenation    = 24
    fused_features = tf.keras.layers.Concatenate(
        name="feature_fusion",
    )(
        [
            local_embedding,
            global_embedding,
        ]
    )

    # Learn interactions between local lesion morphology
    # and global mammographic context.
    x = tf.keras.layers.Dense(
        units=FUSION_UNITS,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizer,
        name="fusion_adapter",
    )(fused_features)

    # LayerNormalization is independent of the fusion batch size.
    x = tf.keras.layers.LayerNormalization(
        name="fusion_adapter_normalization",
    )(x)

    x = tf.keras.layers.ReLU(
        name="fusion_adapter_relu",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=FUSION_DROPOUT,
        name="fusion_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="fusion_output",
    )(x)

    fusion_model = tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=outputs,
        name="local_global_resnet50_fusion",
    )

    return fusion_model


import numpy as np
import tensorflow as tf


def residual_fusion_models(
    local_model,
    global_model,
    local_embedding_layer_name,
    global_embedding_layer_name,
    local_output_layer_name="classification_output",
    correction_units=8,
    correction_dropout=0.3,
    correction_l2=1e-4,
):
    """
    Build a residual local-global fusion model.

    The frozen local classifier provides the baseline logit.
    A small fusion head learns a contextual correction from the
    local and global embeddings.

    Initial prediction:
        fusion prediction == local prediction

    Learned prediction:
        final logit = frozen local logit + contextual correction
    """

    regularizer = tf.keras.regularizers.l2(
        correction_l2
    )

    # ------------------------------------------------------------------
    # Retrieve the required pretrained layers
    # ------------------------------------------------------------------

    try:
        local_embedding_layer = local_model.get_layer(
            local_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local embedding layer "
            f"'{local_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    try:
        global_embedding_layer = global_model.get_layer(
            global_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Global embedding layer "
            f"'{global_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in global_model.layers]}"
        ) from exc

    try:
        local_output_layer = local_model.get_layer(
            local_output_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local output layer "
            f"'{local_output_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    if not isinstance(
        local_output_layer,
        tf.keras.layers.Dense,
    ):
        raise TypeError(
            "The local output layer must be a Dense layer. "
            f"Received: {type(local_output_layer).__name__}"
        )

    if local_output_layer.units != 1:
        raise ValueError(
            "The local output layer must contain one unit. "
            f"Received: {local_output_layer.units}"
        )

    local_activation = tf.keras.activations.serialize(
        local_output_layer.activation
    )

    if local_activation != "sigmoid":
        raise ValueError(
            "The local output layer must use a sigmoid activation. "
            f"Received: {local_activation}"
        )

    # ------------------------------------------------------------------
    # Build frozen branch extractors
    # ------------------------------------------------------------------

    # The local extractor returns:
    #   1. the learned local embedding;
    #   2. the tensor immediately before the final local Dense layer.
    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=[
            local_embedding_layer.output,
            local_output_layer.input,
        ],
        name="local_residual_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_embedding_layer.output,
        name="global_residual_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    print(
        "Local embedding shape:",
        local_extractor.output_shape[0],
    )

    print(
        "Local classifier input shape:",
        local_extractor.output_shape[1],
    )

    print(
        "Global embedding shape:",
        global_extractor.output_shape,
    )

    if len(local_extractor.output_shape[0]) != 2:
        raise ValueError(
            "The local embedding must be flat. "
            f"Received: {local_extractor.output_shape[0]}"
        )

    if len(global_extractor.output_shape) != 2:
        raise ValueError(
            "The global embedding must be flat. "
            f"Received: {global_extractor.output_shape}"
        )

    # ------------------------------------------------------------------
    # Define model inputs
    # ------------------------------------------------------------------

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    # training=False disables branch dropout and augmentation and keeps
    # pretrained BatchNormalization layers in inference mode.
    (
        local_embedding,
        local_classifier_input,
    ) = local_extractor(
        local_input,
        training=False,
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    # ------------------------------------------------------------------
    # Rebuild the frozen pre-sigmoid local classifier
    # ------------------------------------------------------------------

    # Copy the final local Dense weights into a linear frozen layer.
    # This produces the exact local pre-sigmoid logit.
    local_logit_layer = tf.keras.layers.Dense(
        units=1,
        activation=None,
        use_bias=local_output_layer.use_bias,
        trainable=False,
        name="frozen_local_baseline_logit",
    )

    local_logit = local_logit_layer(
        local_classifier_input
    )

    # The new linear layer is now built, so its weights can be copied.
    local_logit_layer.set_weights(
        local_output_layer.get_weights()
    )

    # ------------------------------------------------------------------
    # Normalize branch embeddings independently
    # ------------------------------------------------------------------

    # Keep the normalization configuration that performed better in the
    # standard fusion experiments.
    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    # ------------------------------------------------------------------
    # Contextual correction head
    # ------------------------------------------------------------------

    fused_features = tf.keras.layers.Concatenate(
        name="residual_feature_fusion",
    )(
        [
            local_embedding,
            global_embedding,
        ]
    )

    correction = tf.keras.layers.Dense(
        units=correction_units,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizer,
        name="contextual_correction_adapter",
    )(fused_features)

    correction = tf.keras.layers.LayerNormalization(
        name="contextual_correction_normalization",
    )(correction)

    correction = tf.keras.layers.ReLU(
        name="contextual_correction_relu",
    )(correction)

    correction = tf.keras.layers.Dropout(
        rate=correction_dropout,
        name="contextual_correction_dropout",
    )(correction)

    # Zero initialization guarantees that the initial correction is zero.
    # Therefore, the initial residual model exactly reproduces the local model.
    contextual_logit_correction = tf.keras.layers.Dense(
        units=1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizer,
        name="contextual_logit_correction",
    )(correction)

    final_logit = tf.keras.layers.Add(
        name="local_logit_plus_contextual_correction",
    )(
        [
            local_logit,
            contextual_logit_correction,
        ]
    )

    outputs = tf.keras.layers.Activation(
        activation="sigmoid",
        name="residual_fusion_output",
    )(final_logit)

    residual_model = tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=outputs,
        name="residual_local_global_resnet50_fusion",
    )

    return residual_model

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


# ======================================================================
# Load dataset indexes
# ======================================================================

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

print(
    "\nLocal index:",
    local_index_path,
)

print(
    "Global index:",
    global_index_path,
)

local_df = pd.read_csv(
    local_index_path
)

global_df = pd.read_csv(
    global_index_path
)


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

# fusion_model = fusion_models(
#     local_model=local_model,
#     global_model=global_model,
#     local_embedding_layer_name=LOCAL_EMBEDDING_LAYER,
#     global_embedding_layer_name=GLOBAL_EMBEDDING_LAYER,
# )

fusion_model = residual_fusion_models(
    local_model=local_model,
    global_model=global_model,
    local_embedding_layer_name="mammography_adapter",
    global_embedding_layer_name="mammography_adapter_relu",
    local_output_layer_name="classification_output",
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