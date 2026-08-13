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
    OUTPUT_MODEL,
    OUTPUT_NPY,
    SEED,
)

from src.functions import set_seed

from src.training.dataset_preparation import (
    train_val_test_sets,
)


# ======================================================================
# Configuration
# ======================================================================

TEST_FUSION = True

LOCAL_MODEL_PATH = (
    OUTPUT_MODEL / f"local_resnet50_head.keras"
)

FUSION_MODEL_PATH = (
    OUTPUT_MODEL / f"model_fusion_seed_{SEED}.keras"
)

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
# Initial cleanup
# ======================================================================

tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)


# ======================================================================
# Load the saved fusion model
# ======================================================================

if TEST_FUSION:
    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {FUSION_MODEL_PATH}"
        )

    print("\nLoading fusion model:")
    print(FUSION_MODEL_PATH)

    model = tf.keras.models.load_model(
        FUSION_MODEL_PATH,
        compile=False,
    )

    local_input_shape = model.input_shape[0]
    global_input_shape = model.input_shape[1]

else:
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local checkpoint not found: {LOCAL_MODEL_PATH}"
        )

    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {FUSION_MODEL_PATH}"
        )

    print("\nLoading local model:")
    print(LOCAL_MODEL_PATH)

    model = tf.keras.models.load_model(
        LOCAL_MODEL_PATH,
        compile=False,
    )

    # Shape of the local model itself
    local_input_shape = model.input_shape

    # We still need the global dimensions to reconstruct
    # exactly the paired population used by the fusion model.
    fusion_shape_model = tf.keras.models.load_model(
        FUSION_MODEL_PATH,
        compile=False,
    )

    global_input_shape = (
        fusion_shape_model.input_shape[1]
    )

    del fusion_shape_model
    gc.collect()


local_height = local_input_shape[1]
local_width = local_input_shape[2]

global_height = global_input_shape[1]
global_width = global_input_shape[2]

print(
    "\nLocal input:",
    local_input_shape,
)

print(
    "Global paired input:",
    global_input_shape,
)


# ======================================================================
# Load the local and global dataset indexes
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
    "\nLocal index:"
)

print(
    local_index_path
)

print(
    "\nGlobal index:"
)

print(
    global_index_path
)

if not local_index_path.exists():
    raise FileNotFoundError(
        f"Local dataset index not found: {local_index_path}"
    )

if not global_index_path.exists():
    raise FileNotFoundError(
        f"Global dataset index not found: {global_index_path}"
    )

local_df = pd.read_csv(
    local_index_path
)

global_df = pd.read_csv(
    global_index_path
)


# ======================================================================
# Recreate exactly the same local-global pairing as during training
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
        "Missing columns in local dataframe: "
        f"{missing_local_columns}"
    )

if missing_global_columns:
    raise ValueError(
        "Missing columns in global dataframe: "
        f"{missing_global_columns}"
    )

local_df = local_df.copy()
global_df = global_df.copy()

local_df["local_path"] = local_df[
    "preprocessed_image_path"
]


# Confirm that one mammogram key does not point to several full images.
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
        "Some mammogram keys refer to multiple global image paths:\n"
        f"{conflicting_global_paths.head()}"
    )


global_lookup = (
    global_df[
        MAMMOGRAM_KEY
        + [
            "preprocessed_image_path",
        ]
    ]
    .drop_duplicates(
        subset=MAMMOGRAM_KEY
    )
    .rename(
        columns={
            "preprocessed_image_path": "global_path",
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
    "Successfully paired lesions:",
    len(df),
)

print(
    "Unmatched local lesions:",
    initial_local_count - len(df),
)

if df.empty:
    raise ValueError(
        "No local lesion could be paired with a global mammogram."
    )


# ======================================================================
# Rebuild the TensorFlow datasets
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

# Training and validation datasets are unnecessary for inference.
del train_ds
del val_ds
gc.collect()


print(
    "\nOriginal test dataset specification:"
)

print(
    test_ds.element_spec
)


# Rebatch to one local-global pair at a time.
# This avoids the previous batch-16 memory exhaustion.
test_ds_eval = (
    test_ds
    .unbatch()
    .batch(
        1,
        drop_remainder=False,
    )
    .prefetch(1)
)


# Validate one pair before running the full inference.
if TEST_FUSION:
    for images, labels in test_ds_eval.take(1):
        local_images, global_images = images

        print(
            "\nLocal evaluation batch:",
            local_images.shape,
        )

        print(
            "Global evaluation batch:",
            global_images.shape,
        )

        print(
            "Labels:",
            labels.shape,
        )


# ======================================================================
# Test inference
# ======================================================================

y_true = []
y_prob = []

print(
    "\nRunning test inference..."
)

for batch_number, (images, labels) in enumerate(
    test_ds_eval,
    start=1,
):
    local_images, global_images = images

    if TEST_FUSION:
        probabilities = model(
            [local_images, global_images],
            training=False,
        ).numpy().reshape(-1)

    else:
        probabilities = model(
            local_images,
            training=False,
        ).numpy().reshape(-1)

    y_prob.extend(probabilities)
    y_true.extend(
        labels.numpy().reshape(-1)
    )


y_true = np.asarray(
    y_true,
    dtype=np.int32,
)

y_prob = np.asarray(
    y_prob,
    dtype=np.float32,
)

if len(y_true) == 0:
    raise ValueError(
        "The test dataset produced no samples."
    )

if len(y_true) != len(y_prob):
    raise ValueError(
        "The number of labels and probabilities does not match."
    )


# ======================================================================
# Test metrics
# ======================================================================

model_name = (
    "Fusion"
    if TEST_FUSION
    else "Local-only"
)

print(
    f"\n{model_name} paired-test results:"
)

print(
    f"Number of samples: {len(y_true)}"
)

print(
    f"Minimum probability: {y_prob.min()}, "
    f"maximum probability: {y_prob.max()}, "
    f"average probability: {y_prob.mean()}"
)

auc = roc_auc_score(
    y_true,
    y_prob,
)

ap = average_precision_score(
    y_true,
    y_prob,
)

raw_bce = log_loss(
    y_true,
    y_prob,
    labels=[0, 1],
)

print(
    f"AUC: {auc}"
)

print(
    f"Average Precision (AP): {ap}"
)

print(
    f"Raw binary cross-entropy: {raw_bce}"
)


for threshold in THRESHOLDS:
    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    tp = int(
        (
            (y_pred == 1)
            & (y_true == 1)
        ).sum()
    )

    fp = int(
        (
            (y_pred == 1)
            & (y_true == 0)
        ).sum()
    )

    fn = int(
        (
            (y_pred == 0)
            & (y_true == 1)
        ).sum()
    )

    tn = int(
        (
            (y_pred == 0)
            & (y_true == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(y_true)
    )

    print(
        f"threshold: {threshold:.2f}, "
        f"accuracy: {accuracy}, "
        f"precision: {precision}, "
        f"recall: {recall}"
    )


# ======================================================================
# Save predictions
# ======================================================================

if TEST_FUSION:
    predictions_path = (
        OUTPUT_MODEL
        / f"fusion_test_predictions_seed_{SEED}.csv"
    )
else:
    predictions_path = (
        OUTPUT_MODEL
        / f"local_test_predictions_seed_{SEED}.csv"
    )

predictions_df = pd.DataFrame(
    {
        "true_label": y_true,
        "predicted_probability": y_prob,
    }
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

print(
    "\nPredictions saved to:"
)

print(
    predictions_path
)


# ======================================================================
# Final cleanup
# ======================================================================

del test_ds
del test_ds_eval
del model

tf.keras.backend.clear_session()
gc.collect()