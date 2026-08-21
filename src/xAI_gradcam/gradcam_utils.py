import inspect

import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt


def get_required_layer(model, layer_name):

    try:
        return model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(f"Layer '{layer_name}' was not found in model '{model.name}'. Available layers: {[layer.name for layer in model.layers]}") from exc

def find_nested_resnet50(model):

    candidates = [layer for layer in model.layers if (isinstance(layer, tf.keras.Model) and "resnet50" in layer.name.lower())]

    if len(candidates) != 1:
        raise ValueError(f"Could not identify exactly one nested ResNet50 inside '{model.name}'. Candidates: {[layer.name for layer in candidates]}")

    return candidates[0]


def call_layer_inference(layer, inputs):

    parameters = inspect.signature(layer.call).parameters

    if "training" in parameters:
        return layer(inputs, training=False)

    return layer(inputs)


def apply_layers_inference(inputs, layers):
    
    x = inputs

    for layer in layers:
        x = call_layer_inference(layer, x)

    return x

def build_resnet_feature_model(branch_model, model_name):
    
    resnet = find_nested_resnet50(branch_model)

    gradcam_input = tf.keras.Input(shape=branch_model.input_shape[1:], name=f"{model_name}_input")

    x = gradcam_input

    augmentation_layers = [layer for layer in branch_model.layers if layer.name == "data_augmentation"]

    if len(augmentation_layers) > 1:
        raise ValueError("Several data_augmentation layers were found.")

    if augmentation_layers:
        x = augmentation_layers[0](x, training=False)

    grayscale_to_rgb = (get_required_layer(branch_model, "grayscale_to_rgb"))
    restore_255_scale = (get_required_layer(branch_model, "restore_255_scale"))

    x = grayscale_to_rgb([x, x, x])
    x = restore_255_scale(x)

    x = (tf.keras.applications.resnet50.preprocess_input(x))

    feature_maps = resnet(x, training=False)
    feature_model = tf.keras.Model(inputs=gradcam_input, outputs=feature_maps, name=model_name)

    return feature_model, resnet

def make_gradcam_heatmap(feature_maps, gradients):

    if gradients is None:
        raise RuntimeError("Gradients are None.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(feature_maps * pooled_gradients, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))

    return heatmap.numpy()

def make_branch_gradcam_heatmap(img_array, feature_model, head_layers, output_layer, target_class):

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:

        feature_maps = feature_model(img_tensor, training=False)
        tape.watch(feature_maps)

        x = apply_layers_inference(feature_maps, head_layers)

        # Reconstruct the final Dense layer before sigmoid.
        logit = tf.linalg.matmul(x, output_layer.kernel)

        if output_layer.use_bias:
            logit = tf.nn.bias_add(logit, output_layer.bias)

        # Explain evidence for the requested binary class.
        if target_class == 1:
            target_score = logit[:, 0]
        elif target_class == 0:
            target_score = -logit[:, 0]
        else:
            raise ValueError("target_class must be 0 or 1.")

    gradients = tape.gradient(target_score, feature_maps)

    return make_gradcam_heatmap(feature_maps, gradients)

def load_single_channel_image(image_path, expected_shape):

    image = np.load(image_path).astype(np.float32)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape != tuple(expected_shape):
        raise ValueError(f"Expected image shape {tuple(expected_shape)}, received {image.shape} for '{image_path}'.")

    return image[np.newaxis, ...]


def prepare_image_for_display(image_array):

    image = np.asarray(image_array, dtype=np.float32)

    if image.ndim == 4:
        image = image[0]

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    if image.ndim != 2:
        raise ValueError(f"Expected a single grayscale image, received shape {image.shape}.")

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image_gray = (image - image_min) / (image_max - image_min)
    else:
        image_gray = np.zeros_like(image)

    image_rgb = np.repeat(image_gray[..., np.newaxis], repeats=3, axis=-1)

    return image_gray, image_rgb


def save_image_figure(data, output_path, title, dimensions=(7, 7), cmap=None, vmin=None, vmax=None, contour_data=None, contour_levels=(0.5,), contour_linewidths=2):
    
    plt.figure(figsize=dimensions)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    if contour_data is not None:
        plt.contour(contour_data.astype(np.float32), levels=list(contour_levels), linewidths=contour_linewidths)

    plt.title(title)
    plt.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_gradcam_figures(image_array, heatmap, branch_name, output_prefix, title_details, output_dir, model_name="Grad-CAM", roi_mask=None, heatmap_alpha=0.35):

    image_gray, image_rgb = prepare_image_for_display(image_array)

    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], size=image_gray.shape[:2], method="bilinear").numpy()[..., 0]
    heatmap_resized = np.maximum(heatmap_resized, 0)
    heatmap_resized /= (heatmap_resized.max() + 1e-8)

    colormap = mpl.colormaps["jet"]

    heatmap_rgb = colormap(heatmap_resized)[..., :3]

    overlay = ((1.0 - heatmap_alpha) * image_rgb + heatmap_alpha * heatmap_rgb)
    overlay = np.clip(overlay, 0, 1)

    base_path = output_dir / f"{output_prefix}_{branch_name}"

    original_title = (f"{branch_name.capitalize()} image | {title_details}")
    gradcam_title = (f"{model_name} ({branch_name}) | {title_details}")

    save_image_figure(image_gray, f"{base_path}_original.png", original_title, cmap="gray", vmin=0, vmax=1)

    save_image_figure(heatmap_resized, f"{base_path}_heatmap.png", gradcam_title, cmap="jet", vmin=0, vmax=1)

    save_image_figure(overlay, f"{base_path}_overlay.png", gradcam_title)

    if roi_mask is not None:
        save_image_figure(overlay, f"{base_path}_roi_comparison.png", f"Grad-CAM vs lesion ROI | {title_details}", contour_data=roi_mask)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    figure.suptitle(gradcam_title)
    figure.tight_layout()

    figure.savefig(f"{base_path}_combined.png", dpi=300, bbox_inches="tight")

    plt.close(figure)

    print(f"Saved {branch_name} Grad-CAM figures with prefix: {base_path}")