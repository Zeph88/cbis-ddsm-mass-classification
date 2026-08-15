import inspect

import numpy as np
import tensorflow as tf


def get_required_layer(
    model,
    layer_name,
):
    try:
        return model.get_layer(
            layer_name
        )
    except ValueError as exc:
        raise ValueError(
            f"Layer '{layer_name}' was not found in "
            f"model '{model.name}'. Available layers:\n"
            f"{[layer.name for layer in model.layers]}"
        ) from exc


def find_nested_resnet50(
    model,
):
    candidates = [
        layer
        for layer in model.layers
        if (
            isinstance(
                layer,
                tf.keras.Model,
            )
            and "resnet50"
            in layer.name.lower()
        )
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not identify exactly one nested "
            f"ResNet50 inside '{model.name}'. "
            f"Candidates: "
            f"{[layer.name for layer in candidates]}"
        )

    return candidates[0]


def call_layer_inference(
    layer,
    inputs,
):
    parameters = inspect.signature(
        layer.call
    ).parameters

    if "training" in parameters:
        return layer(
            inputs,
            training=False,
        )

    return layer(inputs)


def apply_layers_inference(
    inputs,
    layers,
):
    x = inputs

    for layer in layers:
        x = call_layer_inference(
            layer,
            x,
        )

    return x

def build_resnet_feature_model(
    branch_model,
    model_name,
):
    resnet = find_nested_resnet50(
        branch_model
    )

    gradcam_input = tf.keras.Input(
        shape=branch_model.input_shape[1:],
        name=f"{model_name}_input",
    )

    x = gradcam_input

    augmentation_layers = [
        layer
        for layer in branch_model.layers
        if layer.name
        == "data_augmentation"
    ]

    if len(augmentation_layers) > 1:
        raise ValueError(
            "Several data_augmentation "
            "layers were found."
        )

    if augmentation_layers:
        x = augmentation_layers[0](
            x,
            training=False,
        )

    grayscale_to_rgb = (
        get_required_layer(
            branch_model,
            "grayscale_to_rgb",
        )
    )

    restore_255_scale = (
        get_required_layer(
            branch_model,
            "restore_255_scale",
        )
    )

    x = grayscale_to_rgb(
        [x, x, x]
    )

    x = restore_255_scale(x)

    x = (
        tf.keras.applications
        .resnet50
        .preprocess_input(x)
    )

    feature_maps = resnet(
        x,
        training=False,
    )

    feature_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=feature_maps,
        name=model_name,
    )

    return feature_model, resnet

def make_gradcam_heatmap(
    feature_maps,
    gradients,
):
    if gradients is None:
        raise RuntimeError(
            "Gradients are None."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps
        * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()

def make_branch_gradcam_heatmap(
    img_array,
    feature_model,
    head_layers,
    output_layer,
    target_class,
):
    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        feature_maps = feature_model(
            img_tensor,
            training=False,
        )

        tape.watch(
            feature_maps
        )

        x = apply_layers_inference(
            feature_maps,
            head_layers,
        )

        # Reconstruct the final Dense layer before sigmoid.
        logit = tf.linalg.matmul(
            x,
            output_layer.kernel,
        )

        if output_layer.use_bias:
            logit = tf.nn.bias_add(
                logit,
                output_layer.bias,
            )

        # Explain evidence for the requested binary class.
        if target_class == 1:
            target_score = logit[:, 0]
        elif target_class == 0:
            target_score = -logit[:, 0]
        else:
            raise ValueError(
                "target_class must be 0 or 1."
            )

    gradients = tape.gradient(
        target_score,
        feature_maps,
    )

    return make_gradcam_heatmap(
        feature_maps=feature_maps,
        gradients=gradients,
    )

def load_single_channel_image(
    image_path,
    expected_shape,
):
    image = np.load(
        image_path
    ).astype(np.float32)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape != tuple(expected_shape):
        raise ValueError(
            f"Expected image shape "
            f"{tuple(expected_shape)}, "
            f"received {image.shape} "
            f"for '{image_path}'."
        )

    return image[np.newaxis, ...]


def prepare_image_for_display(
    image_array,
):
    image = np.asarray(
        image_array,
        dtype=np.float32,
    )

    if image.ndim == 4:
        image = image[0]

    if (
        image.ndim == 3
        and image.shape[-1] == 1
    ):
        image = image[..., 0]

    if image.ndim != 2:
        raise ValueError(
            "Expected a single grayscale image, "
            f"received shape {image.shape}."
        )

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image_gray = (
            image - image_min
        ) / (
            image_max - image_min
        )
    else:
        image_gray = np.zeros_like(
            image
        )

    image_rgb = np.repeat(
        image_gray[..., np.newaxis],
        repeats=3,
        axis=-1,
    )

    return (
        image_gray,
        image_rgb,
    )