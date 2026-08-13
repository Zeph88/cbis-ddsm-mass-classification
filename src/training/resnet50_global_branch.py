import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.training.cnn_evaluation import cnn_predict, evaluate_thresholds
from src.training.dataset_preparation import cnn_steps, train_val_test_sets
from src.functions import set_seed
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_MODEL, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, GLOBAL_HEIGHT, GLOBAL_WIDTH, OUTPUT_PLOT
import math
import gc

tf.keras.backend.clear_session()
set_seed(SEED)

resolution=(GLOBAL_HEIGHT, GLOBAL_WIDTH)

file_path = f"full_{resolution[0]}x{resolution[1]}"

print(f"dataset_index_{file_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{file_path}.csv")

train_ds, val_ds, test_ds = train_val_test_sets(
    df, BATCH_SIZE, SEED
)

for images, labels in train_ds.take(1):
    print("Batch shape:", images.shape)
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())
    print("Mean:", tf.reduce_mean(images).numpy())
    print("Standard deviation:", tf.math.reduce_std(images).numpy())
    print("Labels:", labels.numpy())

del images, labels
gc.collect()

train_steps, val_steps, test_steps = cnn_steps(df)

def build_resnet50_transfer(
    input_shape=(GLOBAL_HEIGHT, GLOBAL_WIDTH, 1),
    dropout_rate=0.5,
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

    # data_augmentation = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.RandomTranslation(
    #             height_factor=0.01,
    #             width_factor=0.01,
    #             fill_mode="constant",
    #             fill_value=0.0,
    #             seed=SEED,
    #             name="random_small_translation",
    #         )
    #     ],
    #     name="data_augmentation",
    # )

    # x = data_augmentation(inputs)

    # input passed 3 times
    x = tf.keras.layers.Concatenate(
        axis=-1,
        name="grayscale_to_rgb",
    )([inputs, inputs, inputs])

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
        pool_size=(6, 6),
        strides=(6, 6),
        padding="same",
        name="resnet_spatial_max_pooling",
    )(x)

    x = tf.keras.layers.Flatten(
        name="resnet_global_flatten",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout_rate,
        name="classification_dropout",
    )(x)

    # Solve dying ReLU issue
    x = tf.keras.layers.Dense(
        units=8,
        activation=None,
        use_bias=False,
        # activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="mammography_adapter",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="mammography_adapter_batch_norm"
    )(x)

    x = tf.keras.layers.ReLU(
        name="mammography_adapter_relu",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.2,
        name="embedding_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        # kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="classification_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="global_resnet50_transfer",
    )

    return model, base_model

model, base_model = build_resnet50_transfer(
    input_shape=(GLOBAL_HEIGHT, GLOBAL_WIDTH, 1),
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

model.summary()

head_checkpoint_path = (
    OUTPUT_MODEL / f"global_resnet50_head.keras"
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
        OUTPUT_MODEL / f"global_resnet50_head.csv"
    ),
]
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_head,
)

history_values = {
    metric_name: list(metric_values)
    for metric_name, metric_values in history_head.history.items()
}

del history_head
del callbacks_head
del model
del base_model

del train_ds
del val_ds
del test_ds

tf.keras.backend.clear_session()
gc.collect()


print("\nHead-only model evaluation:")


model = tf.keras.models.load_model(
    head_checkpoint_path,
    compile=False,
)


EVAL_BATCH_SIZE = 2

eval_train_ds, eval_val_ds, test_ds_eval = train_val_test_sets(
    df,
    EVAL_BATCH_SIZE,
    SEED,
)

del eval_train_ds
del eval_val_ds
gc.collect()


y_prob, y_true = cnn_predict(
    model,
    test_ds_eval,
)

evaluate_thresholds(
    y_prob,
    y_true,
)

del test_ds_eval
del model
tf.keras.backend.clear_session()
gc.collect()


# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(
    history_values["loss"],
    label="Train loss",
)

plt.plot(
    history_values["val_loss"],
    label="Validation loss",
)

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_PLOT
    / (
        "global head only - val_loss and loss - "
        f"{GLOBAL_HEIGHT}x{GLOBAL_WIDTH} - seed {SEED}.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# AUC plot
plt.figure(figsize=(8, 5))

# >>> CHANGED
plt.plot(
    history_values["auc"],
    label="Train AUC",
)

plt.plot(
    history_values["val_auc"],
    label="Validation AUC",
)

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training and validation AUC")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_PLOT
    / (
        "global head only - val_auc and auc - "
        f"{GLOBAL_HEIGHT}x{GLOBAL_WIDTH} - seed {SEED}.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()