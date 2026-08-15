import tensorflow as tf


LOCAL_EMBEDDING_LAYER = "mammography_adapter"
GLOBAL_EMBEDDING_LAYER = "mammography_adapter_relu"
LOCAL_OUTPUT_LAYER = "classification_output"

def build_symmetric_fusion(
    local_model,
    global_model,
    local_embedding_layer_name=LOCAL_EMBEDDING_LAYER,
    global_embedding_layer_name=GLOBAL_EMBEDDING_LAYER,
    fusion_units=16,
    fusion_dropout=0.5,
    fusion_l2=1e-4,
):
    """
    Build a lesion-level fusion model from frozen local and global
    ResNet50 branches.

    The local branch provides lesion morphology.
    The global branch provides mammographic context.
    """

    regularizer = (
        tf.keras.regularizers.l2(
            fusion_l2
        )
    )

    # --------------------------------------------------------------
    # Extract the trained embeddings
    # --------------------------------------------------------------

    try:
        local_embedding_layer = local_model.get_layer(
            local_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local embedding layer "
            f"'{local_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    try:
        global_embedding_layer = global_model.get_layer(
            global_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Global embedding layer "
            f"'{global_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in global_model.layers]}"
        ) from exc

    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=local_embedding_layer.output,
        name="local_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_embedding_layer.output,
        name="global_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    print(
        "Local embedding shape:",
        local_extractor.output_shape,
    )

    print(
        "Global embedding shape:",
        global_extractor.output_shape,
    )

    if len(local_extractor.output_shape) != 2:
        raise ValueError(
            "The local extractor must return a flat embedding. "
            f"Received {local_extractor.output_shape}."
        )

    if len(global_extractor.output_shape) != 2:
        raise ValueError(
            "The global extractor must return a flat embedding. "
            f"Received {global_extractor.output_shape}."
        )

    # --------------------------------------------------------------
    # Define fusion inputs
    # --------------------------------------------------------------

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    # training=False disables branch augmentation and dropout,
    # and keeps their BatchNormalization layers in inference mode.
    local_embedding = local_extractor(
        local_input,
        training=False,
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    # Normalize each representation independently so that one branch
    # does not dominate merely because of a different feature scale.
    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    # Expected dimensions:
    # local embedding  = 16
    # global embedding = 8
    # concatenation    = 24
    fused_features = tf.keras.layers.Concatenate(
        name="feature_fusion",
    )(
        [
            local_embedding,
            global_embedding,
        ]
    )

    # Learn interactions between local lesion morphology
    # and global mammographic context.
    x = tf.keras.layers.Dense(
        units=fusion_units,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizer,
        name="fusion_adapter",
    )(fused_features)

    # LayerNormalization is independent of the fusion batch size.
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

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="fusion_output",
    )(x)

    fusion_model = tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=outputs,
        name="local_global_resnet50_fusion",
    )

    return fusion_model

def build_residual_fusion(
    local_model,
    global_model,
    local_embedding_layer_name=LOCAL_EMBEDDING_LAYER,
    global_embedding_layer_name=GLOBAL_EMBEDDING_LAYER,
    local_output_layer_name=LOCAL_OUTPUT_LAYER,
    correction_units=8,
    correction_dropout=0.3,
    correction_l2=1e-4,
):
    """
    Build a residual local-global fusion model.

    The frozen local classifier provides the baseline logit.
    A small fusion head learns a contextual correction from the
    local and global embeddings.

    Initial prediction:
        fusion prediction == local prediction

    Learned prediction:
        final logit = frozen local logit + contextual correction
    """

    regularizer = tf.keras.regularizers.l2(
        correction_l2
    )

    # ------------------------------------------------------------------
    # Retrieve the required pretrained layers
    # ------------------------------------------------------------------

    try:
        local_embedding_layer = local_model.get_layer(
            local_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local embedding layer "
            f"'{local_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    try:
        global_embedding_layer = global_model.get_layer(
            global_embedding_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Global embedding layer "
            f"'{global_embedding_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in global_model.layers]}"
        ) from exc

    try:
        local_output_layer = local_model.get_layer(
            local_output_layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Local output layer "
            f"'{local_output_layer_name}' was not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in local_model.layers]}"
        ) from exc

    if not isinstance(
        local_output_layer,
        tf.keras.layers.Dense,
    ):
        raise TypeError(
            "The local output layer must be a Dense layer. "
            f"Received: {type(local_output_layer).__name__}"
        )

    if local_output_layer.units != 1:
        raise ValueError(
            "The local output layer must contain one unit. "
            f"Received: {local_output_layer.units}"
        )

    local_activation = tf.keras.activations.serialize(
        local_output_layer.activation
    )

    if local_activation != "sigmoid":
        raise ValueError(
            "The local output layer must use a sigmoid activation. "
            f"Received: {local_activation}"
        )

    # ------------------------------------------------------------------
    # Build frozen branch extractors
    # ------------------------------------------------------------------

    # The local extractor returns:
    #   1. the learned local embedding;
    #   2. the tensor immediately before the final local Dense layer.
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

    print(
        "Local embedding shape:",
        local_extractor.output_shape[0],
    )

    print(
        "Local classifier input shape:",
        local_extractor.output_shape[1],
    )

    print(
        "Global embedding shape:",
        global_extractor.output_shape,
    )

    if len(local_extractor.output_shape[0]) != 2:
        raise ValueError(
            "The local embedding must be flat. "
            f"Received: {local_extractor.output_shape[0]}"
        )

    if len(global_extractor.output_shape) != 2:
        raise ValueError(
            "The global embedding must be flat. "
            f"Received: {global_extractor.output_shape}"
        )

    # ------------------------------------------------------------------
    # Define model inputs
    # ------------------------------------------------------------------

    local_input = tf.keras.Input(
        shape=local_model.input_shape[1:],
        name="local_input",
    )

    global_input = tf.keras.Input(
        shape=global_model.input_shape[1:],
        name="global_input",
    )

    # training=False disables branch dropout and augmentation and keeps
    # pretrained BatchNormalization layers in inference mode.
    (
        local_embedding,
        local_classifier_input,
    ) = local_extractor(
        local_input,
        training=False,
    )

    global_embedding = global_extractor(
        global_input,
        training=False,
    )

    # ------------------------------------------------------------------
    # Rebuild the frozen pre-sigmoid local classifier
    # ------------------------------------------------------------------

    # Copy the final local Dense weights into a linear frozen layer.
    # This produces the exact local pre-sigmoid logit.
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

    # The new linear layer is now built, so its weights can be copied.
    local_logit_layer.set_weights(
        local_output_layer.get_weights()
    )

    # ------------------------------------------------------------------
    # Normalize branch embeddings independently
    # ------------------------------------------------------------------

    # Keep the normalization configuration that performed better in the
    # standard fusion experiments.
    local_embedding = tf.keras.layers.LayerNormalization(
        name="local_embedding_normalization",
    )(local_embedding)

    global_embedding = tf.keras.layers.LayerNormalization(
        name="global_embedding_normalization",
    )(global_embedding)

    # ------------------------------------------------------------------
    # Contextual correction head
    # ------------------------------------------------------------------

    fused_features = tf.keras.layers.Concatenate(
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
        kernel_regularizer=regularizer,
        name="contextual_correction_adapter",
    )(fused_features)

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

    # Zero initialization guarantees that the initial correction is zero.
    # Therefore, the initial residual model exactly reproduces the local model.
    contextual_logit_correction = tf.keras.layers.Dense(
        units=1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizer,
        name="contextual_logit_correction",
    )(correction)

    final_logit = tf.keras.layers.Add(
        name="local_logit_plus_contextual_correction",
    )(
        [
            local_logit,
            contextual_logit_correction,
        ]
    )

    outputs = tf.keras.layers.Activation(
        activation="sigmoid",
        name="residual_fusion_output",
    )(final_logit)

    residual_model = tf.keras.Model(
        inputs=[
            local_input,
            global_input,
        ],
        outputs=outputs,
        name="residual_local_global_resnet50_fusion",
    )

    return residual_model