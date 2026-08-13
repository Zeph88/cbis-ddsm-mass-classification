# https://keras.io/examples/vision/grad_cam/
# Adaptation of code from Keras.io, author F. Chollet

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.functions import ensure_directory

from src.config import (
    OUTPUT_NPY,
    OUTPUT_MODEL,
    OUTPUT_PLOT,
    PIXELS_H,
    PIXELS_W,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

zoom_to_roi = True

if zoom_to_roi:
    image_type = f"zoom_384x384"
else:
    image_type = f"full_{PIXELS_H}x{PIXELS_W}"

print(f"dataset_index_{image_type}.csv")


# ------------------------------------------------------------------
# Load dataset index
# ------------------------------------------------------------------

df = pd.read_csv(
    OUTPUT_NPY / f"dataset_index_{image_type}.csv"
)

# Prefer examples from the test set when the column is available.
if "set" in df.columns:
    sample_df = (
        df[df["set"] == "test"]
        .reset_index(drop=True)
    )
else:
    sample_df = df.reset_index(drop=True)


idx = 35

# iloc avoids relying on the dataframe index labels.
row = sample_df.iloc[idx]

mmg_path = row["preprocessed_image_path"]
true_label = int(row["label"])


# ------------------------------------------------------------------
# Load and prepare image
# ------------------------------------------------------------------

array_npy = np.load(mmg_path).astype(np.float32)

# Ensure the image has a channel dimension.
if array_npy.ndim == 2:
    array_npy = array_npy[..., np.newaxis]

if array_npy.ndim != 3 or array_npy.shape[-1] != 1:
    raise ValueError(
        "Expected one local mammogram with shape (H, W, 1), "
        f"but received {array_npy.shape}."
    )

# Add the batch dimension.
array_npy = array_npy[np.newaxis, ...]

print("Grad-CAM input shape:", array_npy.shape)
print("Input minimum:", array_npy.min())
print("Input maximum:", array_npy.max())


# ------------------------------------------------------------------
# Load local ResNet50 model
# ------------------------------------------------------------------

model = tf.keras.models.load_model(
    OUTPUT_MODEL / "resnet50_head.keras",
    compile=False,
)

# Build the model once.
_ = model(
    array_npy,
    training=False,
)

model.summary()


# ------------------------------------------------------------------
# Find the nested ResNet50 backbone
# ------------------------------------------------------------------

# >>> ADDED
# ResNet50 is a nested Keras model inside the complete local model.
resnet_candidates = [
    layer
    for layer in model.layers
    if (
        isinstance(layer, tf.keras.Model)
        and "resnet50" in layer.name.lower()
    )
]

if len(resnet_candidates) != 1:
    raise ValueError(
        "Could not identify exactly one nested ResNet50 backbone. "
        f"Candidates found: {[layer.name for layer in resnet_candidates]}"
    )

resnet_backbone = resnet_candidates[0]

print("ResNet50 backbone:", resnet_backbone.name)
print("ResNet50 output shape:", resnet_backbone.output_shape)

# ------------------------------------------------------------------
# Reconstruct preprocessing up to the ResNet output
# ------------------------------------------------------------------

# Construct a model that produces the ResNet feature maps from the
# original one-channel mammogram input.
gradcam_input = tf.keras.Input(
    shape=model.input_shape[1:],
    name="gradcam_input",
)

x = gradcam_input

# Apply augmentation only if it exists in the saved model.
augmentation_layers = [
    layer
    for layer in model.layers
    if layer.name == "data_augmentation"
]

if augmentation_layers:
    x = augmentation_layers[0](
        x,
        training=False,
    )


# Reuse the saved preprocessing layers.
grayscale_to_rgb = model.get_layer(
    "grayscale_to_rgb"
)

restore_255_scale = model.get_layer(
    "restore_255_scale"
)

x = grayscale_to_rgb(
    [x, x, x]
)

x = restore_255_scale(x)

x = tf.keras.applications.resnet50.preprocess_input(x)


# ResNet50 include_top=False ends with conv5_block3_out.
feature_maps = resnet_backbone(
    x,
    training=False,
)

feature_model = tf.keras.Model(
    inputs=gradcam_input,
    outputs=feature_maps,
    name="local_resnet50_feature_model",
)


# ------------------------------------------------------------------
# Recover the trained local classification head
# ------------------------------------------------------------------

resnet_position = model.layers.index(
    resnet_backbone
)

output_layer = model.layers[-1]

if not isinstance(output_layer, tf.keras.layers.Dense):
    raise TypeError(
        "The last model layer must be a Dense classification layer, "
        f"but received {type(output_layer).__name__}."
    )

if output_layer.units != 1:
    raise ValueError(
        "The final Dense layer must contain one binary output unit."
    )


# Layers located after ResNet50 and before the final Dense layer.
# For the local branch, this should include:
#
# MaxPooling2D -> Flatten -> Dense16 -> Dropout
head_layers = model.layers[
    resnet_position + 1 : -1
]

print(
    "Local head layers:",
    [layer.name for layer in head_layers],
)

print(
    "Classification output layer:",
    output_layer.name,
)


# ------------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------------

def make_gradcam_heatmap(
    img_array,
    feature_model,
    head_layers,
    output_layer,
):
    """
    Generate a Grad-CAM heatmap from the final ResNet50 feature maps.

    The gradient is computed from the pre-sigmoid malignancy logit
    rather than the probability to reduce sigmoid saturation.
    """

    with tf.GradientTape() as tape:
        # Extract the final convolutional feature maps.
        conv_outputs = feature_model(
            img_array,
            training=False,
        )

        # Explicitly watch the feature maps because the backbone is frozen.
        tape.watch(conv_outputs)

        x = conv_outputs

        # Apply the already-trained local classification head.
        for layer in head_layers:
            x = layer(
                x,
                training=False,
            )

        # >>> ADDED
        # Compute the pre-sigmoid logit manually.
        logits = tf.linalg.matmul(
            x,
            output_layer.kernel,
        )

        if output_layer.use_bias:
            logits = tf.nn.bias_add(
                logits,
                output_layer.bias,
            )

        malignancy_logit = logits[:, 0]

    grads = tape.gradient(
        malignancy_logit,
        conv_outputs,
    )

    if grads is None:
        raise RuntimeError(
            "Gradients are None. Check the reconstructed feature "
            "and classification paths."
        )

    print("Conv outputs:", conv_outputs.shape)
    print("Gradients:", grads.shape)
    print(
        "Malignancy logit:",
        malignancy_logit.numpy(),
    )

    # Average each channel gradient across spatial positions.
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    # Remove the batch dimension.
    conv_outputs = conv_outputs[0]

    # Weight each feature map by its mean gradient.
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1,
    )

    # Keep only positive contributions to the malignant prediction.
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    maximum = tf.reduce_max(heatmap)

    # Avoid division by zero for an empty heatmap.
    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

# >>> CHANGED
# Calling the model directly avoids the extra predict pipeline.
prob = float(
    model(
        array_npy,
        training=False,
    ).numpy()[0, 0]
)

pred_label = int(prob >= 0.5)

true_class = (
    "MALIGNANT"
    if true_label == 1
    else "BENIGN"
)

pred_class = (
    "MALIGNANT"
    if pred_label == 1
    else "BENIGN"
)

print("True label:", true_class)
print("Predicted label:", pred_class)
print("P(malignant):", prob)


# ------------------------------------------------------------------
# Generate heatmap
# ------------------------------------------------------------------

heatmap = make_gradcam_heatmap(
    img_array=array_npy,
    feature_model=feature_model,
    head_layers=head_layers,
    output_layer=output_layer,
)


# ------------------------------------------------------------------
# Prepare visualisations
# ------------------------------------------------------------------

# Remove batch and channel dimensions.
img = array_npy[0, ..., 0].astype(
    np.float32
)

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    size=img.shape[:2],
    method="bilinear",
).numpy()[..., 0]

heatmap_resized = np.maximum(
    heatmap_resized,
    0,
)

heatmap_resized /= (
    heatmap_resized.max() + 1e-8
)


# Ensure the displayed image is in [0, 1].
img_display = img.copy()

img_min = img_display.min()
img_max = img_display.max()

if img_max > img_min:
    img_display = (
        img_display - img_min
    ) / (
        img_max - img_min
    )
else:
    img_display = np.zeros_like(
        img_display
    )


# Convert grayscale image to RGB.
img_rgb = np.repeat(
    img_display[..., np.newaxis],
    repeats=3,
    axis=-1,
)


# Convert heatmap to RGB.
cmap = mpl.colormaps["jet"]

heatmap_rgb = cmap(
    heatmap_resized
)[..., :3]


# Blend image and Grad-CAM.
alpha = 0.35

overlay = (
    (1 - alpha) * img_rgb
    + alpha * heatmap_rgb
)

overlay = np.clip(
    overlay,
    0,
    1,
)


# ------------------------------------------------------------------
# Save figures
# ------------------------------------------------------------------

output_prefix = (
    f"local_gradcam_idx_{idx}"
    f"_true_{true_label}"
    f"_pred_{pred_label}"
)


plt.figure(figsize=(7, 7))

plt.imshow(overlay)

plt.title(
    f"True: {true_class} | "
    f"Pred: {pred_class} | "
    f"P(malignant): {prob:.4f}"
)

plt.axis("off")

plt.savefig(
    OUTPUT_PLOT
    / f"{output_prefix}_overlay.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


plt.figure(figsize=(7, 7))

plt.imshow(
    heatmap_resized,
    cmap="jet",
    vmin=0,
    vmax=1,
)

plt.title(
    f"Local Grad-CAM | "
    f"True: {true_class} | "
    f"Pred: {pred_class}"
)

plt.axis("off")

plt.savefig(
    OUTPUT_PLOT
    / f"{output_prefix}_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


plt.figure(figsize=(7, 7))

plt.imshow(
    img_display,
    cmap="gray",
    vmin=0,
    vmax=1,
)

plt.title(
    f"Original local crop | "
    f"True: {true_class} | "
    f"Pred: {pred_class} | "
    f"P(malignant): {prob:.4f}"
)

plt.axis("off")

plt.savefig(
    OUTPUT_PLOT
    / f"{output_prefix}_original.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()