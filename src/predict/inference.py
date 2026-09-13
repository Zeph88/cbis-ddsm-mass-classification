import os
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import OUTPUT_MODEL, OUTPUT_NPY, PROJECT_ROOT, SEED
from src.data.pairing import pair_local_global
from src.functions import load_json_data, parse_arguments, set_seed, load_data
from src.evaluation.evaluation_utils import validate_model_path


args = parse_arguments(
    description="Run inference on one preprocessed mammography case.",
    arguments=[
        {"name": "--model", "choices": ["local", "global", "symmetric", "residual"], "default": "residual"},
        {"name": "--idx", "type": int, "default": 0}
    ]
)

MODEL_TYPE = args.model
CASE_INDEX = args.idx

MODEL_PATHS = {
    "local": OUTPUT_MODEL / "local_resnet50_head.keras",
    "global": OUTPUT_MODEL / "global_resnet50_head.keras",
    "symmetric": OUTPUT_MODEL / f"model_fusion_symmetric_seed_{SEED}.keras",
    "residual": OUTPUT_MODEL / f"model_fusion_residual_seed_{SEED}.keras",
}

model_path = MODEL_PATHS[MODEL_TYPE]
threshold = (load_json_data(PROJECT_ROOT / f"residual_threshold_seed_{SEED}.json", "selected_threshold") if MODEL_TYPE == "residual" else 0.5)

tf.keras.backend.clear_session()
gc.collect()
set_seed(SEED)
validate_model_path(model_path)

model = tf.keras.models.load_model(model_path, compile=False)

def load_image(path):
    image = np.load(path).astype(np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    return image


def load_index(input_shape, local):
    height, width = int(input_shape[1]), int(input_shape[2])
    name = f"dataset_index_{'zoom' if local else 'full'}_{height}x{width}.csv"
    return pd.read_csv(OUTPUT_NPY / name)


if MODEL_TYPE in {"symmetric", "residual"}:
    local_height, local_width = model.input_shape[0][1:3]
    global_height, global_width = model.input_shape[1][1:3]

    local_path = OUTPUT_NPY / f"dataset_index_zoom_{local_height}x{local_width}.csv"
    global_path = OUTPUT_NPY / f"dataset_index_full_{global_height}x{global_width}.csv"

    local_df, global_df = load_data(local_path, global_path)
    local_df = local_df[local_df["set"] == "test"].copy()
    global_df = global_df[global_df["set"] == "test"].copy()

    df = pair_local_global(local_df, global_df).reset_index(drop=True)

    print("Rows:", len(df))
    print(df.iloc[CASE_INDEX][["local_path", "global_path", "label"]])

    case = df.iloc[CASE_INDEX]

    model_input = [load_image(case["local_path"]), load_image(case["global_path"])]

else:
    local = MODEL_TYPE == "local"
    df = load_index(model.input_shape, local)
    df = df[df["set"] == "test"].reset_index(drop=True)
    case = df.iloc[CASE_INDEX]
    model_input = load_image(case["preprocessed_image_path"])


probability = float(model(model_input, training=False).numpy()[0, 0])
predicted_class = int(probability >= threshold)
true_label = int(case["label"])

labels = {0: "BENIGN", 1: "MALIGNANT"}

print(f"\nModel: {MODEL_TYPE}")
print(f"Case: {CASE_INDEX}")
print(f"Probability: {probability:.4f}")
print(f"Threshold: {threshold:.3f}")
print(f"Prediction: {labels[predicted_class]}")
print(f"Ground truth: {labels[true_label]}")
print(f"Correct: {predicted_class == true_label}")

del model
tf.keras.backend.clear_session()
gc.collect()