import gc
import inspect
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    OUTPUT_MODEL,
    OUTPUT_NPY,
    OUTPUT_PLOT,
    SEED,
    IMAGES_ROOT,
)

from src.preprocessing.dicom_handling import (
    read_dicom_as_array,
)

from src.preprocessing.dataset_preprocessing import (
    orient_by_breast_mass,
)

# ======================================================================
# Configuration
# ======================================================================

# Adjust this path only if the residual checkpoint uses another filename.
RESIDUAL_MODEL_PATH = (
    OUTPUT_MODEL / f"model_fusion_seed_{SEED}.keras"
)

LOCAL_EMBEDDING_LAYER_NAME = "mammography_adapter"
GLOBAL_EMBEDDING_LAYER_NAME = "mammography_adapter_relu"

LOCAL_EXTRACTOR_NAME = "local_residual_feature_extractor"
GLOBAL_EXTRACTOR_NAME = "global_residual_feature_extractor"

LOCAL_BASELINE_LOGIT_LAYER_NAME = "frozen_local_baseline_logit"
LOCAL_NORMALIZATION_LAYER_NAME = "local_embedding_normalization"
GLOBAL_NORMALIZATION_LAYER_NAME = "global_embedding_normalization"
FUSION_LAYER_NAME = "residual_feature_fusion"
CORRECTION_ADAPTER_LAYER_NAME = "contextual_correction_adapter"
CORRECTION_NORMALIZATION_LAYER_NAME = "contextual_correction_normalization"
CORRECTION_RELU_LAYER_NAME = "contextual_correction_relu"
CORRECTION_DROPOUT_LAYER_NAME = "contextual_correction_dropout"
CORRECTION_OUTPUT_LAYER_NAME = "contextual_logit_correction"
FINAL_LOGIT_LAYER_NAME = "local_logit_plus_contextual_correction"
FINAL_OUTPUT_LAYER_NAME = "residual_fusion_output"

MAMMOGRAM_KEY = [
    "patient_id",
    "left or right breast",
    "image view",
]

# Index within the paired test dataframe.
SAMPLE_INDEX = 100

# "predicted" explains the model's predicted class.
# Use 1 to explain malignant evidence or 0 to explain benign evidence.
TARGET_CLASS = 1

DECISION_THRESHOLD = 0.265
HEATMAP_ALPHA = 0.35

GRADCAM_OUTPUT_DIR = (
    OUTPUT_PLOT / "residual_fusion_gradcam"
)


# ======================================================================
# Initial cleanup
# ======================================================================

tf.keras.backend.clear_session()
gc.collect()

GRADCAM_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Utility functions
# ======================================================================


def get_required_layer(model, layer_name):
    """Return one named layer and raise a detailed error if absent."""

    try:
        return model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Layer '{layer_name}' was not found in model "
            f"'{model.name}'. Available layers:\n"
            f"{[layer.name for layer in model.layers]}"
        ) from exc



def find_nested_resnet50(model):
    """Find the single nested ResNet50 backbone in a branch extractor."""

    candidates = [
        layer
        for layer in model.layers
        if (
            isinstance(layer, tf.keras.Model)
            and "resnet50" in layer.name.lower()
        )
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not identify exactly one nested ResNet50 backbone "
            f"inside '{model.name}'. Candidates: "
            f"{[layer.name for layer in candidates]}"
        )

    return candidates[0]



def call_layer_inference(layer, inputs):
    """Call a reused Keras layer in inference mode when supported."""

    call_parameters = inspect.signature(
        layer.call
    ).parameters

    if "training" in call_parameters:
        return layer(
            inputs,
            training=False,
        )

    return layer(inputs)



def apply_layers_inference(inputs, layers):
    """Apply a sequential list of existing layers in inference mode."""

    x = inputs

    for layer in layers:
        x = call_layer_inference(
            layer,
            x,
        )

    return x



def build_resnet_feature_model(
    branch_extractor,
    model_name,
):
    """
    Reconstruct the branch preprocessing path and return the final
    ResNet50 convolutional feature maps.
    """

    resnet_backbone = find_nested_resnet50(
        branch_extractor
    )

    gradcam_input = tf.keras.Input(
        shape=branch_extractor.input_shape[1:],
        name=f"{model_name}_input",
    )

    x = gradcam_input

    augmentation_layers = [
        layer
        for layer in branch_extractor.layers
        if layer.name == "data_augmentation"
    ]

    if len(augmentation_layers) > 1:
        raise ValueError(
            f"Several data_augmentation layers were found in "
            f"'{branch_extractor.name}'."
        )

    if augmentation_layers:
        x = augmentation_layers[0](
            x,
            training=False,
        )

    grayscale_to_rgb = get_required_layer(
        branch_extractor,
        "grayscale_to_rgb",
    )

    restore_255_scale = get_required_layer(
        branch_extractor,
        "restore_255_scale",
    )

    # The saved models duplicate the single grayscale channel to RGB.
    x = grayscale_to_rgb(
        [x, x, x]
    )

    x = restore_255_scale(x)

    x = tf.keras.applications.resnet50.preprocess_input(
        x
    )

    feature_maps = resnet_backbone(
        x,
        training=False,
    )

    feature_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=feature_maps,
        name=model_name,
    )

    return feature_model, resnet_backbone



def get_layers_between(
    model,
    start_layer,
    end_layer,
    include_end=True,
):
    """Return the sequential layers located after start and up to end."""

    start_position = model.layers.index(
        start_layer
    )

    end_position = model.layers.index(
        end_layer
    )

    if end_position <= start_position:
        raise ValueError(
            f"Layer '{end_layer.name}' must occur after "
            f"'{start_layer.name}' in model '{model.name}'."
        )

    stop_position = (
        end_position + 1
        if include_end
        else end_position
    )

    return model.layers[
        start_position + 1 : stop_position
    ]



def load_single_channel_image(
    image_path,
    expected_shape,
):
    """Load one .npy mammogram and add its batch dimension."""

    image = np.load(
        image_path
    ).astype(np.float32)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError(
            "Expected a single-channel image with shape (H, W, 1), "
            f"but received {image.shape} from '{image_path}'."
        )

    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"Image '{image_path}' has shape {image.shape}, but the "
            f"model expects {expected_shape}."
        )

    return image[np.newaxis, ...]



def build_paired_test_dataframe(
    local_index_path,
    global_index_path,
):
    """Pair each local lesion crop with its corresponding mammogram."""

    local_df = pd.read_csv(
        local_index_path
    )

    global_df = pd.read_csv(
        global_index_path
    )

    required_local_columns = (
        MAMMOGRAM_KEY
        + [
            "preprocessed_image_path",
            "resolved_image_file_path",
            "resolved_roi_rel_path",
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

    # Grad-CAM examples are selected from the held-out test subset.
    if "set" in local_df.columns:
        local_df = local_df[
            local_df["set"] == "test"
        ].copy()
    else:
        local_df = local_df.copy()

    if "set" in global_df.columns:
        global_df = global_df[
            global_df["set"] == "test"
        ].copy()
    else:
        global_df = global_df.copy()

    local_df["local_path"] = local_df[
        "preprocessed_image_path"
    ]

    local_df["original_image_path"] = local_df[
        "resolved_image_file_path"
    ]

    local_df["roi_mask_path"] = local_df[
        "resolved_roi_rel_path"
    ]

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
            "Some mammogram keys refer to several global image paths:\n"
            f"{conflicting_global_paths.head()}"
        )

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
            "No local test lesion could be paired with a global image."
        )

    return paired_df.reset_index(
        drop=True
    )



def make_gradcam_heatmap(
    feature_maps,
    gradients,
):
    """Convert final convolutional maps and gradients into Grad-CAM."""

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. Check that the final logit is connected "
            "to the selected branch feature maps."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )

    # Keep positive contributions toward the selected target class.
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()



def prepare_image_for_display(image_array):
    """Convert one batched grayscale image to an RGB display array."""

    image = image_array[
        0,
        ...,
        0,
    ].astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (
            image - minimum
        ) / (
            maximum - minimum
        )
    else:
        image = np.zeros_like(
            image
        )

    image_rgb = np.repeat(
        image[..., np.newaxis],
        repeats=3,
        axis=-1,
    )

    return image, image_rgb



def save_gradcam_figures(
    image_array,
    heatmap,
    branch_name,
    output_prefix,
    title_details,
    roi_mask=None,
):
    """Save original image, heatmap, overlay, and a combined figure."""

    image_gray, image_rgb = prepare_image_for_display(
        image_array
    )

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        size=image_gray.shape[:2],
        method="bilinear",
    ).numpy()[..., 0]

    heatmap_resized = np.maximum(
        heatmap_resized,
        0,
    )

    heatmap_resized /= (
        heatmap_resized.max() + 1e-8
    )

    colormap = mpl.colormaps["jet"]

    heatmap_rgb = colormap(
        heatmap_resized
    )[..., :3]

    overlay = (
        (1.0 - HEATMAP_ALPHA) * image_rgb
        + HEATMAP_ALPHA * heatmap_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        1,
    )

    base_path = (
        GRADCAM_OUTPUT_DIR
        / f"{output_prefix}_{branch_name}"
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(image_gray, cmap="gray", vmin=0, vmax=1)
    plt.title(
        f"{branch_name.capitalize()} image | {title_details}"
    )
    plt.axis("off")
    plt.savefig(
        f"{base_path}_original.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    plt.title(
        f"Residual fusion Grad-CAM ({branch_name}) | "
        f"{title_details}"
    )
    plt.axis("off")
    plt.savefig(
        f"{base_path}_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.title(
        f"Residual fusion Grad-CAM ({branch_name}) | "
        f"{title_details}"
    )
    plt.axis("off")
    plt.savefig(
        f"{base_path}_overlay.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(
        image_gray,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    figure.suptitle(
        f"Residual fusion Grad-CAM ({branch_name}) | "
        f"{title_details}"
    )

    figure.tight_layout()

    figure.savefig(
        f"{base_path}_combined.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved {branch_name} Grad-CAM figures with prefix: "
        f"{base_path}"
    )

    if roi_mask is not None:
        plt.figure(figsize=(7, 7))

        plt.imshow(overlay)

        plt.contour(
            roi_mask.astype(np.float32),
            levels=[0.5],
            linewidths=2,
        )

        plt.title(
            f"Grad-CAM vs lesion ROI | {title_details}"
        )

        plt.axis("off")

        plt.savefig(
            f"{base_path}_roi_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

def evaluate_gradcam_against_roi(
    heatmap,
    roi_mask,
    top_fraction=0.20,
):
    """
    Compare a Grad-CAM heatmap with a binary lesion ROI mask.

    Returns:
        energy_inside_roi
        peak_inside_roi
        dice
        iou
        enrichment
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    roi_mask = np.asarray(
        roi_mask,
        dtype=np.float32,
    )

    if roi_mask.ndim == 3:
        roi_mask = roi_mask[..., 0]

    roi_mask = roi_mask > 0

    # Resize Grad-CAM to exactly the ROI-mask dimensions.
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        size=roi_mask.shape[:2],
        method="bilinear",
    ).numpy()[..., 0]

    heatmap_resized = np.maximum(
        heatmap_resized,
        0,
    )

    total_energy = heatmap_resized.sum()

    if total_energy > 0:
        energy_inside_roi = (
            heatmap_resized[roi_mask].sum()
            / total_energy
        )
    else:
        energy_inside_roi = 0.0

    # Location of maximum Grad-CAM activation.
    peak_position = np.unravel_index(
        np.argmax(heatmap_resized),
        heatmap_resized.shape,
    )

    peak_inside_roi = bool(
        roi_mask[peak_position]
    )

    # Keep the most activated top_fraction of pixels.
    positive_values = heatmap_resized[
        heatmap_resized > 0
    ]

    if positive_values.size > 0:
        threshold = np.quantile(
            positive_values,
            1.0 - top_fraction,
        )

        gradcam_binary = (
            heatmap_resized >= threshold
        )
    else:
        gradcam_binary = np.zeros_like(
            heatmap_resized,
            dtype=bool,
        )

    intersection = np.logical_and(
        gradcam_binary,
        roi_mask,
    ).sum()

    union = np.logical_or(
        gradcam_binary,
        roi_mask,
    ).sum()

    gradcam_pixels = gradcam_binary.sum()
    roi_pixels = roi_mask.sum()

    dice = (
        2.0 * intersection
        / (gradcam_pixels + roi_pixels)
        if gradcam_pixels + roi_pixels > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    roi_area_fraction = (
        roi_pixels / roi_mask.size
    )

    enrichment = (
        energy_inside_roi / roi_area_fraction
        if roi_area_fraction > 0
        else np.nan
    )

    return {
        "energy_inside_roi": float(
            energy_inside_roi
        ),
        "peak_inside_roi": peak_inside_roi,
        "dice_top20": float(dice),
        "iou_top20": float(iou),
        "roi_area_fraction": float(
            roi_area_fraction
        ),
        "enrichment": float(enrichment),
    }

def resize_mask_with_padding(
    mask,
    target_size,
):
    """
    Apply the same aspect-ratio preserving resize and padding
    as the global mammogram preprocessing, using nearest-neighbour
    interpolation to preserve the binary ROI mask.
    """

    mask = tf.convert_to_tensor(
        mask,
        dtype=tf.float32,
    )

    if mask.shape.rank == 2:
        mask = mask[..., tf.newaxis]

    original_height = tf.shape(mask)[0]
    original_width = tf.shape(mask)[1]

    target_height, target_width = target_size

    scale = tf.minimum(
        target_height
        / tf.cast(original_height, tf.float32),
        target_width
        / tf.cast(original_width, tf.float32),
    )

    new_height = tf.cast(
        tf.round(
            tf.cast(original_height, tf.float32)
            * scale
        ),
        tf.int32,
    )

    new_width = tf.cast(
        tf.round(
            tf.cast(original_width, tf.float32)
            * scale
        ),
        tf.int32,
    )

    resized = tf.image.resize(
        mask,
        size=(new_height, new_width),
        method="nearest",
    )

    pad_height = target_height - new_height
    pad_width = target_width - new_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    # Same convention as global mammogram:
    # breast aligned to the right.
    pad_left = pad_width
    pad_right = 0

    padded = tf.pad(
        resized,
        paddings=[
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0],
        ],
        mode="CONSTANT",
        constant_values=0.0,
    )

    return padded

def fix_border_mask(
    mask,
    x_pad,
    y_pad,
    target_size,
):
    """
    Apply the same geometric crop as fix_border(),
    then resize with nearest-neighbour interpolation
    to preserve the binary ROI mask.
    """

    mask = tf.convert_to_tensor(
        mask,
        dtype=tf.float32,
    )

    if mask.shape.rank == 2:
        mask = mask[..., tf.newaxis]

    height = tf.shape(mask)[0]
    width = tf.shape(mask)[1]

    start_x = tf.cast(
        x_pad,
        tf.int32,
    )

    end_x = width

    half_y_pad = y_pad // 2

    start_y = tf.cast(
        half_y_pad,
        tf.int32,
    )

    end_y = (
        height
        - tf.cast(
            half_y_pad,
            tf.int32,
        )
    )

    mask = mask[
        start_y:end_y,
        start_x:end_x,
        :
    ]

    mask = tf.image.resize(
        mask,
        size=target_size,
        method="nearest",
    )

    return mask

# ======================================================================
# Load the residual fusion model
# ======================================================================

residual_model = tf.keras.models.load_model(
    RESIDUAL_MODEL_PATH,
    compile=False,
)

print(
    "Loaded residual model:",
    residual_model.name,
)

if len(residual_model.inputs) != 2:
    raise ValueError(
        "The residual fusion model must have exactly two inputs. "
        f"Received {len(residual_model.inputs)}."
    )

local_input_shape = tuple(
    residual_model.input_shape[0][1:]
)

global_input_shape = tuple(
    residual_model.input_shape[1][1:]
)

print(
    "Local input shape:",
    local_input_shape,
)

print(
    "Global input shape:",
    global_input_shape,
)


# ======================================================================
# Retrieve nested branch extractors and fusion layers
# ======================================================================

local_extractor = get_required_layer(
    residual_model,
    LOCAL_EXTRACTOR_NAME,
)

global_extractor = get_required_layer(
    residual_model,
    GLOBAL_EXTRACTOR_NAME,
)

local_embedding_layer = get_required_layer(
    local_extractor,
    LOCAL_EMBEDDING_LAYER_NAME,
)

global_embedding_layer = get_required_layer(
    global_extractor,
    GLOBAL_EMBEDDING_LAYER_NAME,
)

local_feature_model, local_resnet = build_resnet_feature_model(
    branch_extractor=local_extractor,
    model_name="local_resnet50_gradcam_feature_model",
)

global_feature_model, global_resnet = build_resnet_feature_model(
    branch_extractor=global_extractor,
    model_name="global_resnet50_gradcam_feature_model",
)

local_head_to_embedding = get_layers_between(
    model=local_extractor,
    start_layer=local_resnet,
    end_layer=local_embedding_layer,
    include_end=True,
)

# The local extractor ends at the tensor immediately before the original
# classification Dense layer. Therefore, every remaining layer after the
# embedding belongs to the path toward local_classifier_input.
local_embedding_position = local_extractor.layers.index(
    local_embedding_layer
)

local_embedding_to_classifier_input = local_extractor.layers[
    local_embedding_position + 1 :
]

global_head_to_embedding = get_layers_between(
    model=global_extractor,
    start_layer=global_resnet,
    end_layer=global_embedding_layer,
    include_end=True,
)

print(
    "Local ResNet backbone:",
    local_resnet.name,
)

print(
    "Local head to embedding:",
    [layer.name for layer in local_head_to_embedding],
)

print(
    "Local embedding to classifier input:",
    [
        layer.name
        for layer in local_embedding_to_classifier_input
    ],
)

print(
    "Global ResNet backbone:",
    global_resnet.name,
)

print(
    "Global head to embedding:",
    [layer.name for layer in global_head_to_embedding],
)

local_baseline_logit_layer = get_required_layer(
    residual_model,
    LOCAL_BASELINE_LOGIT_LAYER_NAME,
)

local_normalization_layer = get_required_layer(
    residual_model,
    LOCAL_NORMALIZATION_LAYER_NAME,
)

global_normalization_layer = get_required_layer(
    residual_model,
    GLOBAL_NORMALIZATION_LAYER_NAME,
)

fusion_layer = get_required_layer(
    residual_model,
    FUSION_LAYER_NAME,
)

correction_adapter_layer = get_required_layer(
    residual_model,
    CORRECTION_ADAPTER_LAYER_NAME,
)

correction_normalization_layer = get_required_layer(
    residual_model,
    CORRECTION_NORMALIZATION_LAYER_NAME,
)

correction_relu_layer = get_required_layer(
    residual_model,
    CORRECTION_RELU_LAYER_NAME,
)

correction_dropout_layer = get_required_layer(
    residual_model,
    CORRECTION_DROPOUT_LAYER_NAME,
)

correction_output_layer = get_required_layer(
    residual_model,
    CORRECTION_OUTPUT_LAYER_NAME,
)

final_logit_layer = get_required_layer(
    residual_model,
    FINAL_LOGIT_LAYER_NAME,
)

final_output_layer = get_required_layer(
    residual_model,
    FINAL_OUTPUT_LAYER_NAME,
)


# ======================================================================
# Build the paired test sample
# ======================================================================

local_height, local_width = local_input_shape[:2]
global_height, global_width = global_input_shape[:2]

local_index_path = (
    OUTPUT_NPY
    / f"dataset_index_zoom_{local_height}x{local_width}.csv"
)

global_index_path = (
    OUTPUT_NPY
    / f"dataset_index_full_{global_height}x{global_width}.csv"
)

print(
    "Local index:",
    local_index_path,
)

print(
    "Global index:",
    global_index_path,
)

paired_df = build_paired_test_dataframe(
    local_index_path=local_index_path,
    global_index_path=global_index_path,
)

print(
    "Paired test lesions:",
    len(paired_df),
)

if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(paired_df):
    raise IndexError(
        f"SAMPLE_INDEX={SAMPLE_INDEX} is outside the valid range "
        f"0 to {len(paired_df) - 1}."
    )

row = paired_df.iloc[
    SAMPLE_INDEX
]

original_image_path = (
    IMAGES_ROOT
    / str(row["original_image_path"]).strip()
)

roi_mask_path = (
    IMAGES_ROOT
    / str(row["roi_mask_path"]).strip()
)

print(
    "Original mammogram:",
    original_image_path,
)

print(
    "ROI mask:",
    roi_mask_path,
)

original_image = read_dicom_as_array(
    dicom_path=original_image_path
)

roi_mask = read_dicom_as_array(
    dicom_path=roi_mask_path
)

original_image, roi_mask = orient_by_breast_mass(
    original_image,
    roi_mask,
)

roi_mask = resize_mask_with_padding(
    roi_mask,
    (global_height, global_width),
)

roi_mask = fix_border_mask(
    roi_mask,
    x_pad=100,
    y_pad=20,
    target_size=(
        global_height,
        global_width,
    ),
)

roi_mask = roi_mask.numpy()

if roi_mask.ndim == 3:
    roi_mask = roi_mask[..., 0]

roi_mask = roi_mask > 0.5

local_path = row[
    "local_path"
]

global_path = row[
    "global_path"
]

true_label = int(
    row["label"]
)

local_image = load_single_channel_image(
    image_path=local_path,
    expected_shape=local_input_shape,
)

global_image = load_single_channel_image(
    image_path=global_path,
    expected_shape=global_input_shape,
)

global_spatial_shape = global_image.shape[1:3]

print("ROI mask shape:", roi_mask.shape)
print("Global image shape:", global_spatial_shape)

if tuple(roi_mask.shape) != tuple(global_spatial_shape):
    raise ValueError(
        "ROI mask and preprocessed global mammogram are not "
        f"in the same spatial space: mask={roi_mask.shape}, "
        f"global={global_spatial_shape}"
    )

print(
    "Local image path:",
    local_path,
)

print(
    "Global image path:",
    global_path,
)

print(
    "Local array shape:",
    local_image.shape,
)

print(
    "Global array shape:",
    global_image.shape,
)


# ======================================================================
# Residual fusion Grad-CAM
# ======================================================================

# First obtain the actual model prediction to determine the requested
# target class when TARGET_CLASS == "predicted".
actual_probability = float(
    residual_model(
        [
            local_image,
            global_image,
        ],
        training=False,
    ).numpy()[0, 0]
)

predicted_label = int(
    actual_probability >= DECISION_THRESHOLD
)

if TARGET_CLASS == "predicted":
    target_class = predicted_label
elif TARGET_CLASS in (0, 1):
    target_class = int(TARGET_CLASS)
else:
    raise ValueError(
        "TARGET_CLASS must be 'predicted', 0, or 1."
    )

with tf.GradientTape() as tape:
    # Extract final convolutional feature maps for both branches.
    local_feature_maps = local_feature_model(
        local_image,
        training=False,
    )

    global_feature_maps = global_feature_model(
        global_image,
        training=False,
    )

    # The branch backbones are frozen, so explicitly watch the feature maps.
    tape.watch(
        local_feature_maps
    )

    tape.watch(
        global_feature_maps
    )

    # Reconstruct the local path from ResNet feature maps to the embedding.
    local_embedding = apply_layers_inference(
        inputs=local_feature_maps,
        layers=local_head_to_embedding,
    )

    # Continue from the local embedding to the tensor that originally fed
    # the local sigmoid classification layer.
    local_classifier_input = apply_layers_inference(
        inputs=local_embedding,
        layers=local_embedding_to_classifier_input,
    )

    # Recreate the exact frozen pre-sigmoid local baseline logit.
    local_logit = call_layer_inference(
        local_baseline_logit_layer,
        local_classifier_input,
    )

    # Reconstruct the global path from ResNet feature maps to its embedding.
    global_embedding = apply_layers_inference(
        inputs=global_feature_maps,
        layers=global_head_to_embedding,
    )

    # Apply the trained residual correction head.
    normalized_local_embedding = call_layer_inference(
        local_normalization_layer,
        local_embedding,
    )

    normalized_global_embedding = call_layer_inference(
        global_normalization_layer,
        global_embedding,
    )

    fused_features = fusion_layer(
        [
            normalized_local_embedding,
            normalized_global_embedding,
        ]
    )

    correction = call_layer_inference(
        correction_adapter_layer,
        fused_features,
    )

    correction = call_layer_inference(
        correction_normalization_layer,
        correction,
    )

    correction = call_layer_inference(
        correction_relu_layer,
        correction,
    )

    correction = call_layer_inference(
        correction_dropout_layer,
        correction,
    )

    contextual_logit_correction = call_layer_inference(
        correction_output_layer,
        correction,
    )

    final_logit = final_logit_layer(
        [
            local_logit,
            contextual_logit_correction,
        ]
    )

    manual_probability_tensor = call_layer_inference(
        final_output_layer,
        final_logit,
    )

    # For a malignant explanation, maximize the malignant logit.
    # For a benign explanation, maximize its negative.
    if target_class == 1:
        target_score = final_logit[:, 0]
    else:
        target_score = -final_logit[:, 0]

local_gradients, global_gradients = tape.gradient(
    target_score,
    [
        local_feature_maps,
        global_feature_maps,
    ],
)

manual_probability = float(
    manual_probability_tensor.numpy()[0, 0]
)

local_logit_value = float(
    local_logit.numpy()[0, 0]
)

contextual_correction_value = float(
    contextual_logit_correction.numpy()[0, 0]
)

final_logit_value = float(
    final_logit.numpy()[0, 0]
)

probability_difference = abs(
    actual_probability - manual_probability
)

print(
    "\nResidual fusion diagnostics"
)

print(
    "True label:",
    true_label,
)

print(
    "Predicted label:",
    predicted_label,
)

print(
    "Explained target class:",
    target_class,
)

print(
    "Local baseline logit:",
    local_logit_value,
)

print(
    "Contextual logit correction:",
    contextual_correction_value,
)

print(
    "Final residual logit:",
    final_logit_value,
)

print(
    "Residual probability from loaded model:",
    actual_probability,
)

print(
    "Residual probability from reconstructed path:",
    manual_probability,
)

print(
    "Absolute probability difference:",
    probability_difference,
)

print(
    "Local feature maps:",
    local_feature_maps.shape,
)

print(
    "Global feature maps:",
    global_feature_maps.shape,
)

print(
    "Local gradients:",
    None if local_gradients is None else local_gradients.shape,
)

print(
    "Global gradients:",
    None if global_gradients is None else global_gradients.shape,
)

if probability_difference > 1e-5:
    raise RuntimeError(
        "The manually reconstructed residual path does not reproduce "
        "the loaded model prediction. Check the selected embedding "
        "layers and branch preprocessing."
    )

local_heatmap = make_gradcam_heatmap(
    feature_maps=local_feature_maps,
    gradients=local_gradients,
)

global_heatmap = make_gradcam_heatmap(
    feature_maps=global_feature_maps,
    gradients=global_gradients,
)

global_roi_metrics = evaluate_gradcam_against_roi(
    heatmap=global_heatmap,
    roi_mask=roi_mask,
)

print("\nGlobal Grad-CAM / ROI comparison")

for metric_name, metric_value in global_roi_metrics.items():
    print(
        f"{metric_name}: {metric_value}"
    )

# ======================================================================
# Save local and global explanations
# ======================================================================

true_class_name = (
    "MALIGNANT"
    if true_label == 1
    else "BENIGN"
)

predicted_class_name = (
    "MALIGNANT"
    if predicted_label == 1
    else "BENIGN"
)

target_class_name = (
    "MALIGNANT"
    if target_class == 1
    else "BENIGN"
)

output_prefix = (
    f"residual_fusion_idx_{SAMPLE_INDEX}"
    f"_true_{true_label}"
    f"_pred_{predicted_label}"
    f"_target_{target_class}"
)

title_details = (
    f"True: {true_class_name} | "
    f"Pred: {predicted_class_name} | "
    f"Target: {target_class_name} | "
    f"P(malignant): {actual_probability:.4f} | "
    f"Correction: {contextual_correction_value:+.4f}"
)

save_gradcam_figures(
    image_array=local_image,
    heatmap=local_heatmap,
    branch_name="local",
    output_prefix=output_prefix,
    title_details=title_details,
)

save_gradcam_figures(
    image_array=global_image,
    heatmap=global_heatmap,
    branch_name="global",
    output_prefix=output_prefix,
    title_details=title_details,
    roi_mask=roi_mask,
)

local_probability = float(
    tf.sigmoid(local_logit_value).numpy()
)

local_predicted_label = int(
    local_probability >= DECISION_THRESHOLD
)

correction_changed_decision = (
    local_predicted_label != predicted_label
)

print(
    "Local-only probability:",
    local_probability,
)

print(
    "Local-only predicted label:",
    local_predicted_label,
)

print(
    "Did contextual correction change the decision:",
    correction_changed_decision,
)

# ======================================================================
# Final cleanup
# ======================================================================

del residual_model
del local_feature_model
del global_feature_model

tf.keras.backend.clear_session()
gc.collect()
