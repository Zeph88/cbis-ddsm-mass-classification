# https://keras.io/examples/vision/grad_cam/
# Adaptation of code from Keras.io, author F. Chollet

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.functions import ensure_directory, parse_arguments
from src.config import OUTPUT_NPY, OUTPUT_MODEL, OUTPUT_PLOT, LOCAL_HEIGHT, LOCAL_WIDTH, GLOBAL_HEIGHT, GLOBAL_WIDTH, OPTIMAL_THRESHOLDS
from src.xAI_gradcam.gradcam_utils import build_resnet_feature_model, make_branch_gradcam_heatmap, save_gradcam_figures


ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

args = parse_arguments(
    description=("Generate local/global OOF probabilities from already-trained CV checkpoints."),
    arguments=[
        {
            "name": "--overwrite",
            "action": "store_true",
            "help": ("Overwrite an existing branch_oof_predictions.csv.")
        }
    ],
    exclusive_arguments=[
        {
            "name": "--fold",
            "type": int,
            "choices": range(N_OUTER_FOLDS),
            "help": ("Generate branch OOF predictions for one outer fold.")
        },
        {
            "name": "--all",
            "action": "store_true",
            "help": ("Generate branch OOF predictions for all outer folds.")
        }
    ],
    exclusive_required=True,
)

if args.mode == "local":
    image_type = (f"zoom_{LOCAL_HEIGHT}x{LOCAL_WIDTH}")
    model_path = (OUTPUT_MODEL / "local_resnet50_head.keras")

else:
    image_type = (f"full_{GLOBAL_HEIGHT}x{GLOBAL_WIDTH}")
    model_path = (OUTPUT_MODEL / "global_resnet50_head.keras")

print(f"dataset_index_{image_type}.csv")

# Load dataset index
df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{image_type}.csv")

# Prefer examples from the test set when the column is available.
if "set" in df.columns:
    sample_df = (df[df["set"] == "test"].reset_index(drop=True))
else:
    sample_df = df.reset_index(drop=True)

idx = args.idx

# iloc avoids relying on the dataframe index labels.
row = sample_df.iloc[idx]

mmg_path = row["preprocessed_image_path"]
true_label = int(row["label"])

# Load and prepare image
array_npy = np.load(mmg_path).astype(np.float32)

if array_npy.ndim == 2:
    array_npy = array_npy[..., np.newaxis]

if array_npy.ndim != 3 or array_npy.shape[-1] != 1:
    raise ValueError(f"Expected one mammogram with shape (H, W, 1), but received {array_npy.shape}.")

array_npy = array_npy[np.newaxis, ...]

model = tf.keras.models.load_model(model_path, compile=False)
_ = model(array_npy, training=False)

model.summary()


# Build ResNet50 feature-map extractor
feature_model, resnet_backbone = (build_resnet_feature_model(branch_model=model, model_name=(f"{args.mode}_resnet50_feature_model")))

resnet_position = model.layers.index(resnet_backbone)

output_layer = model.layers[-1]

if not isinstance(output_layer, tf.keras.layers.Dense):
    raise TypeError(f"The last model layer must be a Dense classification layer, but received {type(output_layer).__name__}.")

if output_layer.units != 1:
    raise ValueError("The final Dense layer must contain one binary output unit.")


# Layers located after ResNet50 and before the final Dense layer. For the local branch, this should include.
head_layers = model.layers[resnet_position + 1 : -1]

# Calling the model directly avoids the extra predict pipeline.
prob = float(model(array_npy, training=False).numpy()[0, 0])

pred_label = int(prob >= OPTIMAL_THRESHOLDS)

if args.target_class == "predicted":
    target_class = pred_label
else:
    target_class = int(args.target_class)

target_class_name = ("MALIGNANT" if target_class == 1 else "BENIGN")
true_class = ("MALIGNANT" if true_label == 1 else "BENIGN")
pred_class = ("MALIGNANT" if pred_label == 1 else "BENIGN")

print("True label:", true_class)
print("Predicted label:", pred_class)
print("P(malignant):", prob)


# Generate heatmap
heatmap = make_branch_gradcam_heatmap(
    img_array=array_npy,
    feature_model=feature_model,
    head_layers=head_layers,
    output_layer=output_layer,
    target_class=target_class
)


# Prepare visualisations

img = array_npy[0, ..., 0].astype(np.float32)

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    size=img.shape[:2],
    method="bilinear",
).numpy()[..., 0]

heatmap_resized = np.maximum(heatmap_resized, 0)
heatmap_resized /= (heatmap_resized.max() + 1e-8)

# Ensure the displayed image is in [0, 1].
img_display = img.copy()

img_min = img_display.min()
img_max = img_display.max()

if img_max > img_min:
    img_display = (img_display - img_min) / (img_max - img_min)
else:
    img_display = np.zeros_like(img_display)

# Convert grayscale image to RGB
img_rgb = np.repeat(img_display[..., np.newaxis], repeats=3, axis=-1)

# Convert heatmap to RGB
cmap = mpl.colormaps["jet"]

heatmap_rgb = cmap(heatmap_resized)[..., :3]


# Blend image and Grad-CAM.
alpha = 0.35

overlay = ((1 - alpha) * img_rgb + alpha * heatmap_rgb)

overlay = np.clip(overlay, 0, 1)


# Save figures
output_prefix = (f"{args.mode}_gradcam_idx_{idx}_true_{true_label}_pred_{pred_label}_target_{target_class}")

# Generate heatmap
heatmap = make_branch_gradcam_heatmap(
    img_array=array_npy,
    feature_model=feature_model,
    head_layers=head_layers,
    output_layer=output_layer,
    target_class=target_class,
)

title_details = (f"True: {true_class} | Pred: {pred_class} | P(malignant): {prob:.4f} | Target: {target_class_name}")

save_gradcam_figures(
    image_array=array_npy[0],
    heatmap=heatmap,
    branch_name=args.mode,
    output_prefix=output_prefix,
    title_details=title_details,
    model_name="ResNet50 Grad-CAM",
)

