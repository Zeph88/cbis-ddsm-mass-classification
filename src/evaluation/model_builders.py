import tensorflow as tf

from src.config import (
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LOCAL_HEIGHT,
    LOCAL_WIDTH,
)


LOCAL_EMBEDDING_LAYER = "mammography_adapter"
GLOBAL_EMBEDDING_LAYER = "mammography_adapter_relu"
LOCAL_OUTPUT_LAYER = "classification_output"


def build_local_model(
    seed=42,
    input_shape=(LOCAL_HEIGHT, LOCAL_WIDTH, 1),
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=seed,
                name="random_horizontal_flip",
            )
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs)

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
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-5
        ),
        name=LOCAL_EMBEDDING_LAYER,
    )(x)

    x = tf.keras.layers.Dropout(
        0.5,
        name="classification_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-5
        ),
        name=LOCAL_OUTPUT_LAYER,
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="local_resnet50_transfer",
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
        name=GLOBAL_EMBEDDING_LAYER,
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

def build_symmetric_fusion(
    local_model,
    global_model,
    fusion_units=16,
    fusion_dropout=0.5,
    fusion_l2=1e-4,
):
    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=local_model.get_layer(
            LOCAL_EMBEDDING_LAYER
        ).output,
        name="local_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_model.get_layer(
            GLOBAL_EMBEDDING_LAYER
        ).output,
        name="global_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    local_embedding = local_extractor(
        local_input,
        training=False,
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    fused = tf.keras.layers.Concatenate(
        name="feature_fusion",
    )(
        [
            local_embedding,
            global_embedding,
        ]
    )

    x = tf.keras.layers.Dense(
        units=fusion_units,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            fusion_l2
        ),
        name="fusion_adapter",
    )(fused)

    x = tf.keras.layers.LayerNormalization(
        name="fusion_adapter_normalization",
    )(x)

    x = tf.keras.layers.ReLU(
        name="fusion_adapter_relu",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=fusion_dropout,
        name="fusion_dropout",
    )(x)

    output = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="fusion_output",
    )(x)

    return tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=output,
        name="local_global_resnet50_fusion",
    )

def build_residual_fusion(
    local_model,
    global_model,
    correction_units=8,
    correction_dropout=0.3,
    correction_l2=1e-4,
):
    local_embedding_layer = local_model.get_layer(
        LOCAL_EMBEDDING_LAYER
    )

    global_embedding_layer = global_model.get_layer(
        GLOBAL_EMBEDDING_LAYER
    )

    local_output_layer = local_model.get_layer(
        LOCAL_OUTPUT_LAYER
    )

    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=[
            local_embedding_layer.output,
            local_output_layer.input,
        ],
        name="local_residual_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_embedding_layer.output,
        name="global_residual_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    local_embedding, local_classifier_input = (
        local_extractor(
            local_input,
            training=False,
        )
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    # Reconstruction du logit local pré-sigmoïde.
    local_logit_layer = tf.keras.layers.Dense(
        units=1,
        activation=None,
        use_bias=local_output_layer.use_bias,
        trainable=False,
        name="frozen_local_baseline_logit",
    )

    local_logit = local_logit_layer(
        local_classifier_input
    )

    local_logit_layer.set_weights(
        local_output_layer.get_weights()
    )

    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    fused = tf.keras.layers.Concatenate(
        name="residual_feature_fusion",
    )(
        [
            local_embedding,
            global_embedding,
        ]
    )

    correction = tf.keras.layers.Dense(
        units=correction_units,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            correction_l2
        ),
        name="contextual_correction_adapter",
    )(fused)

    correction = tf.keras.layers.LayerNormalization(
        name="contextual_correction_normalization",
    )(correction)

    correction = tf.keras.layers.ReLU(
        name="contextual_correction_relu",
    )(correction)

    correction = tf.keras.layers.Dropout(
        rate=correction_dropout,
        name="contextual_correction_dropout",
    )(correction)

    delta_logit = tf.keras.layers.Dense(
        units=1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.l2(
            correction_l2
        ),
        name="contextual_logit_correction",
    )(correction)

    final_logit = tf.keras.layers.Add(
        name="local_logit_plus_contextual_correction",
    )(
        [
            local_logit,
            delta_logit,
        ]
    )

    output = tf.keras.layers.Activation(
        activation="sigmoid",
        name="residual_fusion_output",
    )(final_logit)

    return tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=output,
        name="residual_local_global_resnet50_fusion",
    )