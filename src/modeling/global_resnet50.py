import tensorflow as tf

from src.config import (
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
)

def build_global_model(
    seed=42,
    input_shape=(GLOBAL_HEIGHT, GLOBAL_WIDTH, 1),
):
    del seed

    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

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
        input_shape=(
            input_shape[0],
            input_shape[1],
            3,
        ),
    )

    base_model.trainable = False

    x = base_model(
        x,
        training=False,
    )

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
        0.5,
        name="classification_dropout",
    )(x)

    x = tf.keras.layers.Dense(
        units=8,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-4
        ),
        name="mammography_adapter",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="mammography_adapter_batch_norm",
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
        name="classification_output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="global_resnet50_transfer",
    )

