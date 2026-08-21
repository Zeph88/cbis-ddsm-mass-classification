import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.training.dataset_preparation import cnn_steps, train_val_test_sets
from src.functions import set_seed, ensure_directory
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_MODEL, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, LOCAL_HEIGHT, LOCAL_WIDTH, OUTPUT_PLOT
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions, build_binary_metrics, plot_training_metric
from src.training.training_utils import callbacks_for
import math

from src.modeling.local_resnet50 import (
    build_local_model,
)

set_seed(SEED)

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

zoom_to_roi=True
resolution=(LOCAL_HEIGHT, LOCAL_WIDTH)


if zoom_to_roi:
    file_path = f"zoom_{resolution[0]}x{resolution[1]}"
else:
    file_path = f"full_{resolution[0]}x{resolution[1]}"

print(f"dataset_index_{file_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{file_path}.csv")

train_ds, val_ds, test_ds = train_val_test_sets(
    df, BATCH_SIZE, SEED
)

train_steps, val_steps, test_steps = cnn_steps(df)

model = build_local_model(
    input_shape=(
        LOCAL_HEIGHT,
        LOCAL_WIDTH,
        1,
    ),
    seed=SEED,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=build_binary_metrics()
)

head_checkpoint_path = (
    OUTPUT_MODEL / f"local_resnet50_head.keras"
)

callbacks_head = callbacks_for(head_checkpoint_path, OUTPUT_MODEL / f"local_resnet50_head.csv")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_head,
)

print("\nHead-only model evaluation:")

model = tf.keras.models.load_model(
    head_checkpoint_path,
    compile=False,
)

y_true, y_prob = collect_binary_predictions(model, test_ds)

for threshold in [0.35, 0.4, 0.45, 0.5]:
    metrics = calculate_metrics(y_true, y_prob, threshold)
    print(f"threshold : {threshold}, accuracy : {metrics["accuracy"]}, precision : {metrics["precision"]}, recall : {metrics["recall"]}")

# Loss plot
plot_training_metric(
    history=history_head,
    metric_name="loss",
    ylabel="Binary cross-entropy",
    title="Training and validation loss",
    output_path=(OUTPUT_PLOT / f"head only - val_loss and loss - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png")
)

# AUC plot
plot_training_metric(
    history=history_head,
    metric_name="auc",
    ylabel="AUC",
    title="Training and validation AUC",
    output_path=(OUTPUT_PLOT / f"head only - val_auc and auc - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png")
)