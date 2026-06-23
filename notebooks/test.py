import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.functions import train_val_test_sets, set_seed, cnn_predict, evaluate_thresholds, cnn_steps
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, PIXELS_H, PIXELS_W
import math

set_seed(SEED)

zoom_to_roi=True
zoom_margin=30
mask_mode="soft"
factor=0.3


if zoom_to_roi:
    zoom_path = "zoom" + str(zoom_margin)
else:
    zoom_path = "full"

if mask_mode == "soft":
    mask_path = mask_mode + str(factor)
elif mask_mode == "emphasis":
    mask_path = mask_mode + str(factor)
elif mask_mode == "hard":
    mask_path = mask_mode
else:
    mask_path = "nomask"

print(f"dataset_index_{mask_path}_{zoom_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{mask_path}_{zoom_path}.csv")
df = df[df["preprocessed_image_path"]!="N/A"]

print(df["preprocessed_image_path"].head())
print(df["preprocessed_image_path"].dtype)
print(df["label"].head())
print(df["label"].dtype)

print(df["preprocessed_image_path"].isna().sum())
print(df["label"].isna().sum())

print(df["preprocessed_image_path"].map(type).value_counts())
print(df["label"].map(type).value_counts())