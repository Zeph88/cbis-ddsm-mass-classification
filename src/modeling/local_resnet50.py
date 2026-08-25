import tensorflow as tf

from src.config import (
    LOCAL_HEIGHT,
    LOCAL_WIDTH,
)


def build_local_model(
    seed=42,
    input_shape=(
        LOCAL_HEIGHT,
        LOCAL_WIDTH,
        1
    )
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=seed,
                name="random_horizontal_flip",
            )
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = tf.keras.layers.Concatenate(
        axis=-1,
        name="grayscale_to_rgb",
    )([x, x, x])

    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name="restore_255_scale",
    )(x)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = (
        tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(
                input_shape[0],
                input_shape[1],
                3,
            ),
        )
    )

    base_model.trainable = False

    x = base_model(
        x,
        training=False
    )

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding="same",
        name="resnet_channel_max_pooling",
    )(x)

    x = tf.keras.layers.Flatten(
        name="resnet_global_flatten",
    )(x)

    x = tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_regularizer=(
            tf.keras.regularizers.l2(
                1e-5
            )
        ),
        name="mammography_adapter"
    )(x)

    x = tf.keras.layers.Dropout(
        0.5,
        name="classification_dropout"
    )(x)

    output = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_regularizer=(
            tf.keras.regularizers.l2(
                1e-5
            )
        ),
        name="classification_output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="local_resnet50_transfer",
    )