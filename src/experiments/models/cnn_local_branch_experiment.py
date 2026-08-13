import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.training.cnn_evaluation import cnn_predict, evaluate_thresholds
from src.training.dataset_preparation import cnn_steps, train_val_test_sets
from src.functions import set_seed, ensure_directory
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_MODEL, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, LOCAL_HEIGHT, LOCAL_WIDTH, OUTPUT_PLOT
import math

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


def build_baseline_cnn(input_shape=(598, 598, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
        )(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return tf.keras.Model(inputs, outputs)

model = build_baseline_cnn(input_shape=(LOCAL_HEIGHT, LOCAL_WIDTH, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall_35", thresholds=0.35),
        tf.keras.metrics.Precision(name="precision_35", thresholds=0.35),
        tf.keras.metrics.Recall(name="recall_40", thresholds=0.40),
        tf.keras.metrics.Precision(name="precision_40", thresholds=0.40),
        tf.keras.metrics.Recall(name="recall_45", thresholds=0.45),
        tf.keras.metrics.Precision(name="precision_45", thresholds=0.45),
        tf.keras.metrics.Recall(name="recall_50", thresholds=0.50),
        tf.keras.metrics.Precision(name="precision_50", thresholds=0.50),
    ]
)

model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    OUTPUT_MODEL / "model_local_branch.keras",
    monitor="val_loss",
    mode="min",
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

model = tf.keras.models.load_model(OUTPUT_MODEL / "model_local_branch.keras")

y_prob, y_true = cnn_predict(model, test_ds)
evaluate_thresholds(y_prob, y_true)

# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT / f"val_loss and loss - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png", dpi=300, bbox_inches="tight")

# AUC plot
plt.figure(figsize=(8, 5))

plt.plot(history.history["auc"], label="Train AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training and validation AUC")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT / f"val_auc and auc - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png", dpi=300, bbox_inches="tight")