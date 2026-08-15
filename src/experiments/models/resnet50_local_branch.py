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



def build_resnet50_transfer(
    input_shape=(LOCAL_HEIGHT, LOCAL_WIDTH, 1),
    dropout_rate=0.5,
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=SEED,
                name="random_horizontal_flip",
            ),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    # input passed 3 times
    x = tf.keras.layers.Concatenate(
        axis=-1,
        name="grayscale_to_rgb",
    )([x, x, x])

    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name="restore_255_scale",
    )(x)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
    )

    base_model.trainable = False

    # training=False keeps BatchNormalization in inference mode
    x = base_model(x, training=False)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding="same",
        name="resnet_channel_max_pooling",
    )(x)

    # x = tf.keras.layers.SpatialDropout2D(
    #     rate=0.10,
    #     name="resnet_spatial_dropout",
    # )(x)

    x = tf.keras.layers.Flatten(
        name="resnet_global_flatten",
    )(x)

    x = tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="mammography_adapter",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout_rate,
        name="classification_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="classification_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="local_resnet50_transfer",
    )

    return model, base_model

model, base_model = build_resnet50_transfer(
    input_shape=(LOCAL_HEIGHT, LOCAL_WIDTH, 1),
    dropout_rate=0.5,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC",
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
        tf.keras.metrics.Recall(
            name="recall_40",
            thresholds=0.40,
        ),
        tf.keras.metrics.Precision(
            name="precision_40",
            thresholds=0.40,
        ),
        tf.keras.metrics.Recall(
            name="recall_45",
            thresholds=0.45,
        ),
        tf.keras.metrics.Precision(
            name="precision_45",
            thresholds=0.45,
        ),
        tf.keras.metrics.Recall(
            name="recall_50",
            thresholds=0.50,
        ),
        tf.keras.metrics.Precision(
            name="precision_50",
            thresholds=0.50,
        ),
        tf.keras.metrics.Recall(
            name="recall_55",
            thresholds=0.55,
        ),
        tf.keras.metrics.Precision(
            name="precision_55",
            thresholds=0.55,
        ),
    ],
)

head_checkpoint_path = (
    OUTPUT_MODEL / f"local_resnet50_head.keras"
)

callbacks_head = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        restore_best_weights=False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        head_checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.CSVLogger(
        OUTPUT_MODEL / f"local_resnet50_head.csv"
    ),
]

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

y_prob, y_true = cnn_predict(model, test_ds)
evaluate_thresholds(y_prob, y_true)

# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(history_head.history["loss"], label="Train loss")
plt.plot(history_head.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT / f"head only - val_loss and loss - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png", dpi=300, bbox_inches="tight")
plt.close()

# AUC plot
plt.figure(figsize=(8, 5))

plt.plot(history_head.history["auc"], label="Train AUC")
plt.plot(history_head.history["val_auc"], label="Validation AUC")

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training and validation AUC")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT / f"head only - val_auc and auc - {LOCAL_HEIGHT}x{LOCAL_WIDTH} - seed {SEED}.png", dpi=300, bbox_inches="tight")
plt.close()

