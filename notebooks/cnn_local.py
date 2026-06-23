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
factor = 0.7


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

train_ds, val_ds, test_ds = train_val_test_sets(
    df, BATCH_SIZE, SEED
)

train_steps, val_steps, test_steps = cnn_steps(df)


def build_baseline_cnn(input_shape=(256, 256, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model

model = build_baseline_cnn(input_shape=(PIXELS_H, PIXELS_W, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall_40", thresholds=0.40),
        tf.keras.metrics.Precision(name="precision_40", thresholds=0.40),
        tf.keras.metrics.Recall(name="recall_50", thresholds=0.50),
        tf.keras.metrics.Precision(name="precision_50", thresholds=0.50),
    ]
)

model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model_local_three_nodes.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True
)

for x, y in train_ds.take(1):
    print(x.shape)
    print(y.shape)

history = model.fit(
    train_ds.repeat(),
    validation_data=val_ds.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)

model = tf.keras.models.load_model("model_local_three_nodes.keras")

y_prob, y_true = cnn_predict(model, test_ds)
evaluate_thresholds(y_prob, y_true)