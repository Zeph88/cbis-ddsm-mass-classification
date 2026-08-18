import os
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_auc_score,
)

from src.config import OUTPUT_MODEL, OUTPUT_NPY, OUTPUT_PLOT, SEED, MAMMOGRAM_KEY

from src.functions import set_seed, ensure_directory

from src.training.dataset_preparation import (
    train_val_test_sets,
)


# ======================================================================
# Configuration
# ======================================================================

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

RETAINED_THRESHOLD = 0.265

# Available values:
#   "local"
#   "global"
#   "fusion"
BRANCH = "fusion"

# Available values:
#   "paired": evaluate on the local-global fusion subset
#   "native": evaluate the standalone branch on its original dataset
#
# Recommended combinations:
#   local  + paired  -> fair comparison with fusion
#   local  + native  -> original local performance
#   global + native  -> original global performance
#   fusion + paired  -> fusion performance
EVALUATION_SCOPE = "paired"


LOCAL_MODEL_PATH = (
    OUTPUT_MODEL / "local_resnet50_head.keras"
)

GLOBAL_MODEL_PATH = (
    OUTPUT_MODEL / "global_resnet50_head.keras"
)

FUSION_MODEL_PATH = (
    OUTPUT_MODEL / f"model_fusion_seed_{SEED}.keras"
)


MODEL_PATHS = {
    "local": LOCAL_MODEL_PATH,
    "global": GLOBAL_MODEL_PATH,
    "fusion": FUSION_MODEL_PATH,
}


THRESHOLDS = [
    RETAINED_THRESHOLD,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


EVALUATION_BATCH_SIZE = 1


# ======================================================================
# Configuration validation
# ======================================================================

VALID_BRANCHES = {
    "local",
    "global",
    "fusion",
}

VALID_SCOPES = {
    "native",
    "paired",
}


BRANCH = BRANCH.strip().lower()
EVALUATION_SCOPE = EVALUATION_SCOPE.strip().lower()


if BRANCH not in VALID_BRANCHES:
    raise ValueError(
        f"Invalid branch: {BRANCH}. "
        f"Expected one of {sorted(VALID_BRANCHES)}."
    )


if EVALUATION_SCOPE not in VALID_SCOPES:
    raise ValueError(
        f"Invalid evaluation scope: {EVALUATION_SCOPE}. "
        f"Expected one of {sorted(VALID_SCOPES)}."
    )


if BRANCH == "fusion" and EVALUATION_SCOPE != "paired":
    raise ValueError(
        "The fusion model requires EVALUATION_SCOPE='paired'."
    )


if BRANCH == "global" and EVALUATION_SCOPE == "paired":
    print(
        "\nWARNING:"
        "\nThe global branch will be evaluated against local lesion labels."
        "\nThis measures its relevance to lesion classification, not its"
        "\noriginal mammogram-level performance."
    )


# ======================================================================
# Initial cleanup
# ======================================================================

tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)


# ======================================================================
# Model and shape utilities
# ======================================================================

def validate_model_path(model_path):
    """
    Validate that a saved Keras model exists.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )


def validate_single_input_shape(input_shape, model_name):
    """
    Validate and return a single image input shape.
    """

    if not isinstance(input_shape, tuple):
        raise ValueError(
            f"{model_name} should have one input. "
            f"Received: {input_shape}"
        )

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected {model_name} input shape: {input_shape}"
        )

    return tuple(input_shape)


def inspect_single_model_input_shape(model_path):
    """
    Load a model temporarily and return its input shape.

    The temporary model is removed before the prediction model is loaded.
    """

    validate_model_path(model_path)

    temporary_model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    input_shape = validate_single_input_shape(
        temporary_model.input_shape,
        model_path.name,
    )

    del temporary_model

    tf.keras.backend.clear_session()
    gc.collect()

    return input_shape


def get_image_dimensions(input_shape):
    """
    Extract image height and width from a Keras input shape.
    """

    return (
        int(input_shape[1]),
        int(input_shape[2]),
    )


# ======================================================================
# Determine the required input shapes
# ======================================================================

model_path = MODEL_PATHS[BRANCH]

validate_model_path(model_path)


if BRANCH == "fusion":
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if (
        not isinstance(model.input_shape, list)
        or len(model.input_shape) != 2
    ):
        raise ValueError(
            "The fusion model should have two inputs. "
            f"Received: {model.input_shape}"
        )

    local_input_shape = tuple(
        model.input_shape[0]
    )

    global_input_shape = tuple(
        model.input_shape[1]
    )


elif BRANCH == "local":
    if EVALUATION_SCOPE == "paired":
        # The global shape is needed to rebuild the paired dataset.
        global_input_shape = inspect_single_model_input_shape(
            GLOBAL_MODEL_PATH
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    local_input_shape = validate_single_input_shape(
        model.input_shape,
        "local model",
    )


elif BRANCH == "global":
    if EVALUATION_SCOPE == "paired":
        # The local shape is needed to rebuild the paired dataset.
        local_input_shape = inspect_single_model_input_shape(
            LOCAL_MODEL_PATH
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    global_input_shape = validate_single_input_shape(
        model.input_shape,
        "global model",
    )


print(
    f"\nLoaded branch: {BRANCH}"
)

print(
    f"Evaluation scope: {EVALUATION_SCOPE}"
)

print(
    f"Model path: {model_path}"
)


if BRANCH in {"local", "fusion"}:
    print(
        "Local input shape:",
        local_input_shape,
    )


if BRANCH in {"global", "fusion"}:
    print(
        "Global input shape:",
        global_input_shape,
    )


# ======================================================================
# Index loading utilities
# ======================================================================

def load_local_index(input_shape):
    """
    Load the local crop index matching the model input resolution.
    """

    height, width = get_image_dimensions(
        input_shape
    )

    index_path = (
        OUTPUT_NPY
        / f"dataset_index_zoom_{height}x{width}.csv"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Local dataset index not found: {index_path}"
        )

    print(
        "\nLocal index:",
        index_path,
    )

    return pd.read_csv(
        index_path
    )


def load_global_index(input_shape):
    """
    Load the global mammogram index matching the model input resolution.
    """

    height, width = get_image_dimensions(
        input_shape
    )

    index_path = (
        OUTPUT_NPY
        / f"dataset_index_full_{height}x{width}.csv"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Global dataset index not found: {index_path}"
        )

    print(
        "\nGlobal index:",
        index_path,
    )

    return pd.read_csv(
        index_path
    )


# ======================================================================
# Dataset preparation utilities
# ======================================================================

def validate_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """
    Validate the columns required for prediction.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {dataframe_name}: "
            f"{missing_columns}"
        )


def build_paired_dataframe(
    local_dataframe,
    global_dataframe,
):
    """
    Pair every local lesion with its corresponding full mammogram.

    The target remains the label of the local crop.
    """

    local_dataframe = local_dataframe.copy()
    global_dataframe = global_dataframe.copy()

    validate_columns(
        local_dataframe,
        MAMMOGRAM_KEY
        + [
            "preprocessed_image_path",
            "label",
        ],
        "local dataframe",
    )

    validate_columns(
        global_dataframe,
        MAMMOGRAM_KEY
        + [
            "preprocessed_image_path",
        ],
        "global dataframe",
    )

    local_dataframe["local_path"] = local_dataframe[
        "preprocessed_image_path"
    ]

    # Ensure that one mammogram identity does not point to
    # several different global files.
    global_path_count = (
        global_dataframe
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
            "Some mammogram keys refer to several global paths:\n"
            f"{conflicting_global_paths.head()}"
        )

    global_lookup = (
        global_dataframe[
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
        local_dataframe
    )

    paired_dataframe = local_dataframe.merge(
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
        len(paired_dataframe),
    )

    print(
        "Unmatched local lesions:",
        initial_local_count - len(paired_dataframe),
    )

    if paired_dataframe.empty:
        raise ValueError(
            "No local lesion could be paired with a global mammogram."
        )

    return paired_dataframe


def build_native_dataset(
    dataframe,
    input_shape,
):
    """
    Build a dataset for one standalone branch.
    """

    dataframe = dataframe.copy()

    validate_columns(
        dataframe,
        [
            "preprocessed_image_path",
            "label",
        ],
        f"{BRANCH} dataframe",
    )

    dataframe["prediction_path"] = dataframe[
        "preprocessed_image_path"
    ]

    height, width = get_image_dimensions(
        input_shape
    )

    return train_val_test_sets(
        dataframe,
        path_image="prediction_path",
        image_height=height,
        image_width=width,
    )


def build_paired_dataset(
    paired_dataframe,
    local_shape,
    global_shape,
):
    """
    Build a dataset returning local-global image pairs.
    """

    local_height, local_width = get_image_dimensions(
        local_shape
    )

    global_height, global_width = get_image_dimensions(
        global_shape
    )

    return train_val_test_sets(
        paired_dataframe,
        path_image="local_path",
        added_path_image="global_path",
        image_height=local_height,
        image_width=local_width,
        added_image_height=global_height,
        added_image_width=global_width,
    )


# ======================================================================
# Build the requested dataset
# ======================================================================

if EVALUATION_SCOPE == "paired":
    local_df = load_local_index(
        local_input_shape
    )

    global_df = load_global_index(
        global_input_shape
    )

    prediction_df = build_paired_dataframe(
        local_df,
        global_df,
    )

    train_ds, val_ds, test_ds = build_paired_dataset(
        prediction_df,
        local_input_shape,
        global_input_shape,
    )


elif BRANCH == "local":
    local_df = load_local_index(
        local_input_shape
    )

    train_ds, val_ds, test_ds = build_native_dataset(
        local_df,
        local_input_shape,
    )


elif BRANCH == "global":
    global_df = load_global_index(
        global_input_shape
    )

    train_ds, val_ds, test_ds = build_native_dataset(
        global_df,
        global_input_shape,
    )


else:
    raise RuntimeError(
        "Unsupported dataset configuration."
    )


# Training and validation datasets are unnecessary for prediction.
del train_ds
del val_ds

gc.collect()


# ======================================================================
# Rebatch for memory-safe inference
# ======================================================================

test_ds_eval = (
    test_ds
    .unbatch()
    .batch(
        EVALUATION_BATCH_SIZE,
        drop_remainder=False,
    )
    .prefetch(1)
)


print(
    "\nTest dataset specification:"
)

print(
    test_ds_eval.element_spec
)


# ======================================================================
# Select the model input according to the requested branch
# ======================================================================

def select_model_input(images):
    """
    Select the appropriate tensor from a paired dataset.

    Native datasets already contain a single input tensor.
    """

    if EVALUATION_SCOPE == "native":
        return images

    local_images, global_images = images

    if BRANCH == "local":
        return local_images

    if BRANCH == "global":
        return global_images

    if BRANCH == "fusion":
        return [
            local_images,
            global_images,
        ]

    raise ValueError(
        f"Unsupported branch: {BRANCH}"
    )


# ======================================================================
# Validate one batch
# ======================================================================

for images, labels in test_ds_eval.take(1):
    model_inputs = select_model_input(
        images
    )

    if BRANCH == "fusion":
        print(
            "\nLocal evaluation batch:",
            model_inputs[0].shape,
        )

        print(
            "Global evaluation batch:",
            model_inputs[1].shape,
        )

    else:
        print(
            f"\n{BRANCH.capitalize()} evaluation batch:",
            model_inputs.shape,
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
    f"\nRunning {BRANCH} test inference..."
)


for batch_number, (images, labels) in enumerate(
    test_ds_eval,
    start=1,
):
    model_inputs = select_model_input(
        images
    )

    probabilities = model(
        model_inputs,
        training=False,
    ).numpy().reshape(-1)

    y_prob.extend(
        probabilities
    )

    y_true.extend(
        labels.numpy().reshape(-1)
    )

    if batch_number % 50 == 0:
        print(
            f"Processed {batch_number} test samples"
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
        "The test dataset produced no observations."
    )


if len(y_true) != len(y_prob):
    raise ValueError(
        "The numbers of labels and probabilities differ."
    )


# ======================================================================
# Test metrics
# ======================================================================

print(
    f"\n{BRANCH.capitalize()} test results "
    f"({EVALUATION_SCOPE} evaluation):"
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

pr_auc = average_precision_score(
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
    f"PR-AUC: {pr_auc}"
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

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(y_true)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    print(
        f"threshold: {threshold:.3f}, "
        f"accuracy: {accuracy:.4f}, "
        f"precision: {precision:.4f}, "
        f"recall: {recall:.4f}, "
        f"specificity: {specificity:.4f}, "
        f"f1: {f1:.4f}, "
        f"balanced_accuracy: {balanced_accuracy:.4f}, "
        f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
    )

# ======================================================================
# Confusion matrix at the retained threshold
# ======================================================================

y_pred_retained = (
    y_prob >= RETAINED_THRESHOLD
).astype(np.int32)

cm = confusion_matrix(
    y_true,
    y_pred_retained,
    labels=[0, 1],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign",
        "Malignant",
    ],
)

fig, ax = plt.subplots(
    figsize=(5, 5)
)

display.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title(
    f"Confusion Matrix – Test Set\n"
    f"Threshold = {RETAINED_THRESHOLD:.3f}"
)

fig.tight_layout()

confusion_matrix_path = (
    OUTPUT_PLOT
    / (
        f"confusion_matrix_{BRANCH}_"
        f"{EVALUATION_SCOPE}_"
        f"seed_{SEED}.png"
    )
)

fig.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nConfusion matrix saved to:"
)
print(
    confusion_matrix_path
)

# ======================================================================
# Save predictions
# ======================================================================

predictions_path = (
    OUTPUT_MODEL
    / (
        f"{BRANCH}_{EVALUATION_SCOPE}"
        f"_test_predictions_seed_{SEED}.csv"
    )
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