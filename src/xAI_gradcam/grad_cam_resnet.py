# https://keras.io/examples/vision/grad_cam/
# Adaptation of code from Keras.io, author F. Chollet

import numpy as np
import pandas as pd
import tensorflow as tf
from src.functions import ensure_directory, parse_arguments
from src.config import OUTPUT_NPY, OUTPUT_PLOT, LOCAL_HEIGHT, LOCAL_WIDTH, GLOBAL_HEIGHT, GLOBAL_WIDTH, PROJECT_ROOT, OUTPUT_MODEL
from src.xAI_gradcam.gradcam_utils import build_resnet_feature_model, make_branch_gradcam_heatmap, get_layers_between, save_gradcam_figures

ensure_directory(OUTPUT_PLOT)
PREDICTION_THRESHOLD = 0.5

args = parse_arguments(
    description=("Apply Grad-CAM to a local or global ResNet50 mammography model."),
    arguments=[
        {
            "name": "--mode",
            "choices": ["local", "global"],
            "required": True,
        },
        {
            "name": "--idx",
            "type": int,
            "required": True,
        },
        {
            "name": "--target_class",
            "choices": ["predicted", "0", "1"],
            "default": "predicted",
        }
    ]
)

if args.mode == "local":
    image_type = f"zoom_{LOCAL_HEIGHT}x{LOCAL_WIDTH}"
    model_path = OUTPUT_MODEL / "local_resnet50_head.keras"

else:
    image_type = f"full_{GLOBAL_HEIGHT}x{GLOBAL_WIDTH}"
    model_path = OUTPUT_MODEL / "global_resnet50_head.keras"

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

# Build ResNet50 feature-map extractor
feature_model, resnet_backbone = build_resnet_feature_model(model, f"{args.mode}_resnet50_feature_model")

output_layer = model.layers[-1]

if output_layer.units != 1:
    raise ValueError("The final Dense layer must contain one binary output unit.")

head_layers = get_layers_between(model=model, start_layer=resnet_backbone, end_layer=output_layer, include_end=False)


# Calling the model directly avoids the extra predict pipeline.
prob = float(model(array_npy, training=False).numpy()[0, 0])

pred_label = int(prob >= PREDICTION_THRESHOLD)

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
    target_class=target_class,
)

output_prefix = (f"{args.mode}_gradcam_idx_{idx}_true_{true_label}_pred_{pred_label}_target_{target_class}")
title_details = (f"True: {true_class} | Pred: {pred_class} | P(malignant): {prob:.4f} | Target: {target_class_name}")

save_gradcam_figures(
    image_array=array_npy[0],
    heatmap=heatmap,
    branch_name=args.mode,
    output_prefix=output_prefix,
    title_details=title_details,
    output_dir=OUTPUT_PLOT,
    model_name="ResNet50 Grad-CAM"
)

