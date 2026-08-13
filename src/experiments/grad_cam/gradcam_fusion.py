import numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_NPY, OUTPUT_MODEL


param_inputs = {
    "zoom": {
        "width": 256,
        "height": 256,
    },
    "full": {
        "width": 512,
        "height": 768,
    },
}


# ------------------------------------------------------------------
# local/global pair preparation
# ------------------------------------------------------------------

local_df = pd.read_csv(OUTPUT_NPY / (f"dataset_index_zoom_{param_inputs['zoom']['height']}x{param_inputs['zoom']['width']}.csv"))
global_df = pd.read_csv(OUTPUT_NPY / (f"dataset_index_full_{param_inputs['full']['height']}x{param_inputs['full']['width']}.csv"))

local_df = local_df.copy()
global_df = global_df.copy()

local_df["local_path"] = local_df["preprocessed_image_path"]
global_df["global_path"] = global_df["preprocessed_image_path"]

df = pd.merge(local_df, global_df[["lesion_key", "global_path"]], on="lesion_key", how="inner", validate="one_to_one")

idx = 41
row = df.iloc[idx]

local_path = row["local_path"]
global_path = row["global_path"]
true_label = int(row["label"])


def load_image_batch(path):
    image = np.load(path).astype("float32")

    if image.ndim == 2:
        image = image[..., np.newaxis]

    return image[np.newaxis, ...]


local_array = load_image_batch(local_path)
global_array = load_image_batch(global_path)

print("Local :", local_array.shape)
print("Global:", global_array.shape)

gmic = tf.keras.models.load_model(OUTPUT_MODEL / "model_fusion.keras")

_ = gmic([local_array, global_array], training=False)

print("GMIC input shape :", gmic.input_shape)
print("GMIC output shape :", gmic.output_shape)

local_last_conv_layer_name = "conv2d_2"
global_last_conv_layer_name = "conv2d_2"


def make_gradcam_heatmap(local_array, global_array, model, branch, last_conv_layer_name, target_class=1):
    
    local_extractor = model.get_layer("local_feature_extractor")
    global_extractor = model.get_layer("global_feature_extractor")

    local_input, global_input = model.inputs

    if branch == "local":
        conv_layer = local_extractor.get_layer(last_conv_layer_name)

        cam_extractor = keras.Model(
            inputs=local_extractor.input,
            outputs=[
                conv_layer.output,
                local_extractor.output,
            ],
            name="local_cam_extractor"
        )

        conv_outputs, local_features = cam_extractor(local_input, training=False)
        global_features = global_extractor(global_input, training=False)

    elif branch == "global":
        conv_layer = global_extractor.get_layer(last_conv_layer_name)

        cam_extractor = keras.Model(
            inputs=global_extractor.input,
            outputs=[
                conv_layer.output,
                global_extractor.output,
            ],
            name="global_cam_extractor",
        )

        conv_outputs, global_features = cam_extractor(global_input, training=False)
        local_features = local_extractor(local_input, training=False)

    else:
        raise ValueError("branch should either be 'local' or 'global'.")

    
    # ====================================================================================
    # related to Regularization - change of dim (2nd improvement)
    # ====================================================================================


    global_features = model.get_layer("global_normalization")(global_features) 
    local_features = model.get_layer("local_normalization")(local_features)
    global_features = model.get_layer("global_projection")(global_features)  
    local_features = model.get_layer("local_projection")(local_features)

    # ====================================================================================
    # related to Regularization - change of dim (2nd improvement)
    # ====================================================================================

    fused_features = model.get_layer("feature_fusion")([global_features, local_features])
    x = model.get_layer("fusion_dropout")(fused_features, training=False)

    predictions = model.get_layer("fusion_output")(x)
    grad_model = keras.Model(inputs=[local_input, global_input], outputs=[conv_outputs, predictions])

    with tf.GradientTape() as tape:
        conv_outputs_value, predictions_value = grad_model([local_array, global_array], training=False)

        if target_class == 1:
            class_score = predictions_value[:, 0]
        else:
            class_score = 1.0 - predictions_value[:, 0]

    gradients = tape.gradient(class_score, conv_outputs_value)

    if gradients is None:
        raise RuntimeError(
            f"Impossible to compute gradient for the {branch} branch."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs_value = conv_outputs_value[0]

    heatmap = tf.reduce_sum(conv_outputs_value * pooled_gradients[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    probability = float(predictions_value[0, 0])

    return heatmap.numpy(), probability


local_heatmap, probability = make_gradcam_heatmap(
    local_array=local_array,
    global_array=global_array,
    model=gmic,
    branch="local",
    last_conv_layer_name=local_last_conv_layer_name,
    target_class=1,
)

global_heatmap, _ = make_gradcam_heatmap(
    local_array=local_array,
    global_array=global_array,
    model=gmic,
    branch="global",
    last_conv_layer_name=global_last_conv_layer_name,
    target_class=1,
)


predicted_label = int(probability >= 0.5)
true_class = ("MALIGNANT" if true_label == 1 else "BENIGN")
predicted_class = ("MALIGNANT" if predicted_label == 1 else "BENIGN")

print(f"True label: {true_class}")
print(f"Predicted label: {predicted_class}")
print(f"P(malignant): {probability:.4f}")


def save_gradcam_overlay(
    image_batch,
    heatmap,
    output_path,
    title,
    alpha=0.35,
):
    image = np.squeeze(image_batch, axis=0)

    if image.shape[-1] == 1:
        image = image[..., 0]

    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[0], image.shape[1])).numpy().squeeze()
    heatmap_resized = np.maximum(heatmap_resized, 0)
    heatmap_resized /= (heatmap_resized.max() + 1e-8)

    display_image = image.copy()

    if (display_image.min() < 0 or display_image.max() > 1):
        
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)

    image_rgb = np.stack([display_image, display_image, display_image], axis=-1)
    colormap = mpl.colormaps["jet"]
    heatmap_rgb = colormap(heatmap_resized)[..., :3]

    overlay = ((1 - alpha) * image_rgb + alpha * heatmap_rgb)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


common_title = (f"True: {true_class} | Pred: {predicted_class} | P(malignant): {probability:.4f}")

save_gradcam_overlay(image_batch=local_array, heatmap=local_heatmap, output_path="gradcam_local_overlay.png", title=f"Local Grad-CAM | {common_title}")
save_gradcam_overlay(image_batch=global_array, heatmap=global_heatmap, output_path="gradcam_global_overlay.png", title=f"Global Grad-CAM | {common_title}")