import os
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_auc_score

from src.config import OUTPUT_MODEL, OUTPUT_NPY, OUTPUT_PLOT, SEED, MAMMOGRAM_KEY, THRESHOLDS
from src.data.pairing import pair_local_global, validate_columns
from src.functions import set_seed, ensure_directory, parse_arguments, load_json_data
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions, validate_model_path
from src.training.dataset_preparation import train_val_test_sets

# Recommended combinations:
#   local  + paired  -> fair comparison with fusion
#   local  + native  -> original local performance
#   global + native  -> original global performance
#   fusion + paired  -> fusion performance

args = parse_arguments(
    description="Evaluate a saved model.",
    arguments=[
        {
            "name": "--model",
            "choices": [
                "local",
                "global",
                "symmetric",
                "residual",
            ],
            "required": True,
        },
        {
            "name": "--scope",
            "choices": [
                "native",
                "paired",
            ],
            "default": "paired",
        },
    ],
)

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

OPTIMAL_THRESHOLDS = load_json_data(f"residual_thresholds_seed_{SEED}.json", "selected_threshold")
MODEL_TYPE = args.model
BRANCH = ("fusion" if MODEL_TYPE in {"symmetric", "residual"} else MODEL_TYPE)
EVALUATION_SCOPE = args.scope

LOCAL_MODEL_PATH = (OUTPUT_MODEL / "local_resnet50_head.keras")
GLOBAL_MODEL_PATH = (OUTPUT_MODEL / "global_resnet50_head.keras")
SYMMETRIC_FUSION_MODEL_PATH = (OUTPUT_MODEL / f"model_fusion_symmetric_seed_{SEED}.keras")
RESIDUAL_FUSION_MODEL_PATH = (OUTPUT_MODEL / f"model_fusion_residual_seed_{SEED}.keras")

MODEL_PATHS = {
    "local": LOCAL_MODEL_PATH,
    "global": GLOBAL_MODEL_PATH,
    "symmetric": SYMMETRIC_FUSION_MODEL_PATH,
    "residual": RESIDUAL_FUSION_MODEL_PATH
}

model_path = MODEL_PATHS[MODEL_TYPE]
ALL_THRESHOLDS = THRESHOLDS + [OPTIMAL_THRESHOLDS]

EVALUATION_BATCH_SIZE = 1


# Configuration validation
VALID_BRANCHES = {
    "local",
    "global",
    "fusion"
}

VALID_SCOPES = {
    "native",
    "paired"
}


BRANCH = BRANCH.strip().lower()
EVALUATION_SCOPE = EVALUATION_SCOPE.strip().lower()


if BRANCH not in VALID_BRANCHES:
    raise ValueError(f"Invalid branch: {BRANCH}. Expected one of {sorted(VALID_BRANCHES)}.")

if EVALUATION_SCOPE not in VALID_SCOPES:
    raise ValueError(f"Invalid evaluation scope: {EVALUATION_SCOPE}. Expected one of {sorted(VALID_SCOPES)}.")

if BRANCH == "fusion" and EVALUATION_SCOPE != "paired":
    raise ValueError("The fusion model requires EVALUATION_SCOPE='paired'.")

if BRANCH == "global" and EVALUATION_SCOPE == "paired":
    print("\nWARNING: The global branch will be evaluated against local lesion labels. This measures its relevance to lesion classification, not its noriginal mammogram-level performance.")

# Initial cleanup
tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)


def validate_single_input_shape(input_shape, model_name):

    if not isinstance(input_shape, tuple):
        raise ValueError(f"{model_name} should have one input. Received: {input_shape}")

    if len(input_shape) != 4:
        raise ValueError(f"Unexpected {model_name} input shape: {input_shape}")

    return tuple(input_shape)


def inspect_single_model_input_shape(model_path):

    validate_model_path(model_path)
    temporary_model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = validate_single_input_shape(temporary_model.input_shape, model_path.name)

    del temporary_model

    tf.keras.backend.clear_session()
    gc.collect()

    return input_shape


def get_image_dimensions(input_shape):
    return int(input_shape[1]), int(input_shape[2])


# Determine the required input shapes
validate_model_path(model_path)


if BRANCH == "fusion":
    model = tf.keras.models.load_model(model_path, compile=False)

    if not isinstance(model.input_shape, list) or len(model.input_shape) != 2:
        raise ValueError(f"The fusion model should have two inputs. Received: {model.input_shape}")

    local_input_shape = tuple(model.input_shape[0])
    global_input_shape = tuple(model.input_shape[1])

elif BRANCH == "local":
    if EVALUATION_SCOPE == "paired":
        global_input_shape = inspect_single_model_input_shape(GLOBAL_MODEL_PATH)

    model = tf.keras.models.load_model(model_path, compile=False)
    local_input_shape = validate_single_input_shape(model.input_shape, "local model")


elif BRANCH == "global":
    if EVALUATION_SCOPE == "paired":
        local_input_shape = inspect_single_model_input_shape(LOCAL_MODEL_PATH)

    model = tf.keras.models.load_model(model_path, compile=False)
    global_input_shape = validate_single_input_shape(model.input_shape, "global model")


# Index loading utilities
def load_index(input_shape, zoom_to_roi):
    height, width = get_image_dimensions(input_shape)
    if zoom_to_roi:
        index_path = OUTPUT_NPY / f"dataset_index_zoom_{height}x{width}.csv"
        word = "Local"
    else:
        index_path = (OUTPUT_NPY / f"dataset_index_full_{height}x{width}.csv")
        word = "Global"

    if not index_path.exists():
        raise FileNotFoundError(f"{word} dataset index not found: {index_path}")
    return pd.read_csv(index_path)


# Dataset preparation utilities
def build_paired_dataframe(local_dataframe, global_dataframe):

    validate_columns(local_dataframe, ["label"], "local dataframe")
    initial_local_count = len(local_dataframe)
    paired_dataframe = pair_local_global(local_dataframe, global_dataframe)

    return paired_dataframe


def build_native_dataset(dataframe, input_shape):

    dataframe = dataframe.copy()

    validate_columns(dataframe, ["preprocessed_image_path", "label"], f"{BRANCH} dataframe")
    dataframe["prediction_path"] = dataframe["preprocessed_image_path"]
    height, width = get_image_dimensions(input_shape)

    return train_val_test_sets(
        dataframe,
        path_image="prediction_path",
        image_height=height,
        image_width=width,
    )


def build_paired_dataset(paired_dataframe, local_shape, global_shape):
    
    local_height, local_width = get_image_dimensions(local_shape)
    global_height, global_width = get_image_dimensions(global_shape)

    return train_val_test_sets(
        paired_dataframe,
        path_image="local_path",
        added_path_image="global_path",
        image_height=local_height,
        image_width=local_width,
        added_image_height=global_height,
        added_image_width=global_width,
    )

# Build the requested dataset
if EVALUATION_SCOPE == "paired":
    local_df = load_index(local_input_shape, True)
    global_df = load_index(global_input_shape, False)
    prediction_df = build_paired_dataframe(local_df, global_df)
    train_ds, val_ds, test_ds = build_paired_dataset(prediction_df, local_input_shape, global_input_shape)

elif BRANCH == "local":
    local_df = load_index(local_input_shape, True)
    train_ds, val_ds, test_ds = build_native_dataset(local_df, local_input_shape)

elif BRANCH == "global":
    global_df = load_index(global_input_shape, False)
    train_ds, val_ds, test_ds = build_native_dataset(global_df, global_input_shape)

else:
    raise RuntimeError("Unsupported dataset configuration.")


# Training and validation datasets are unnecessary for prediction.
del train_ds
del val_ds

gc.collect()


# Rebatch for memory-safe inference
test_ds_eval = test_ds.unbatch().batch(EVALUATION_BATCH_SIZE, drop_remainder=False).prefetch(1)

print("\nTest dataset specification:")
print(test_ds_eval.element_spec)

# Select the model input according to the requested branch
def select_model_input(images):
    
    if EVALUATION_SCOPE == "native":
        return images

    local_images, global_images = images

    if BRANCH == "local":
        return local_images

    if BRANCH == "global":
        return global_images

    if BRANCH == "fusion":
        return [local_images, global_images]

    raise ValueError(f"Unsupported branch: {BRANCH}")

# Validate one batch
for images, labels in test_ds_eval.take(1):
    model_inputs = select_model_input(images)

    if BRANCH == "fusion":
        print("\nLocal evaluation batch:", model_inputs[0].shape)
        print("Global evaluation batch:", model_inputs[1].shape)
    else:
        print(f"\n{BRANCH.capitalize()} evaluation batch:", model_inputs.shape)
    
    print("Labels:", labels.shape)

# Inference
y_true, y_prob = collect_binary_predictions(model=model, dataset=test_ds_eval, input_selector=select_model_input)
metrics = calculate_metrics(y_true, y_prob)

print(f"\n{BRANCH.capitalize()} test results ({EVALUATION_SCOPE} evaluation):")
print(f"Probability range: {y_prob.min():.4f}–{y_prob.max():.4f}, mean: {y_prob.mean():.4f}")
print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['ap']:.4f}, BCE: {metrics['bce']:.4f}, Brier: {metrics['brier']:.4f}")

for threshold in ALL_THRESHOLDS:    
    metrics = calculate_metrics(y_true, y_prob, threshold)
    print(
        f"threshold: {threshold:.3f}, "
        f"accuracy: {metrics["accuracy"]:.4f}, "
        f"precision: {metrics["precision"]:.4f}, "
        f"recall: {metrics["recall"]:.4f}, "
        f"specificity: {metrics["specificity"]:.4f}, "
        f"f1: {metrics["f1"]:.4f}, "
        f"balanced_accuracy: {metrics["balanced_accuracy"]:.4f}, "
        f"TP: {metrics["tp"]}, TN: {metrics["tn"]}, FP: {metrics["fp"]}, FN: {metrics["fn"]}"
    )

# Confusion matrix at the retained threshold
y_pred_retained = (y_prob >= OPTIMAL_THRESHOLDS).astype(np.int32)
cm = confusion_matrix(y_true, y_pred_retained, labels=[0, 1])
display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(5, 5))
display.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix - Test Set Threshold = {OPTIMAL_THRESHOLDS:.3f}")
fig.tight_layout()

confusion_matrix_path = (OUTPUT_PLOT / f"confusion_matrix_{BRANCH}_{EVALUATION_SCOPE}_seed_{SEED}.png")
fig.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")

plt.close(fig)

print(f"Confusion matrix saved to {confusion_matrix_path}")

# Save predictions
predictions_path = OUTPUT_MODEL / f"{BRANCH}_{EVALUATION_SCOPE}_test_predictions_seed_{SEED}.csv"

predictions_df = pd.DataFrame(
    {
        "true_label": y_true,
        "predicted_probability": y_prob,
    }
)


predictions_df.to_csv(predictions_path, index=False)

print(f"Predictions saved to {predictions_path}")


# Final cleanup
del test_ds
del test_ds_eval
del model

tf.keras.backend.clear_session()
gc.collect()