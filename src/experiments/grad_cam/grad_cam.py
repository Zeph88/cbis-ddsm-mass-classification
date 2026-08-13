# https://keras.io/examples/vision/grad_cam/
# Adaptation of a code from Keras.io, author fchollet

import os
from src.preprocessing.dataset_preprocessing import tensor_to_2d_np, orient_by_breast_mass, crop_zoom_to_roi
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY, PIXELS_H, PIXELS_W, OUTPUT_MODEL
from src.functions import set_seed, ensure_directory

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display 
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ensure_directory(OUTPUT_MODEL)

zoom_to_roi=True


if zoom_to_roi:
    zoom_path = "zoom_" + str(PIXELS_H) + "x" + str(PIXELS_W)
else:
    zoom_path = "full_" + str(PIXELS_H) + "x" + str(PIXELS_W)

print(f"dataset_index_{zoom_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{zoom_path}.csv")
idx = 2
mmg_path = df["preprocessed_image_path"][idx]
true_label = df["label"][idx]

array_npy = np.load(mmg_path)
array_npy = np.expand_dims(array_npy, axis=0)

model = tf.keras.models.load_model(OUTPUT_MODEL / "model_local_branch.keras")
_ = model(array_npy)

last_conv_layer_name = "conv2d_2"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    print("conv_outputs:", last_conv_layer_output.shape)
    print("predictions:", preds.shape)
    print("class_channel:", class_channel)
    print("grads:", grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Predict
preds = model.predict(array_npy)
prob = float(preds[0][0])
pred_label = int(prob >= 0.5)

true_class = "MALIGNANT" if true_label == 1 else "BENIGN"
pred_class = "MALIGNANT" if pred_label == 1 else "BENIGN"

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(array_npy, model, last_conv_layer_name)

# Original image: remove batch/channel dimensions
img = np.squeeze(array_npy).astype("float32")

# Resize heatmap properly to image size
heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (img.shape[0], img.shape[1])
).numpy().squeeze()

# Normalize heatmap
heatmap_resized = np.maximum(heatmap_resized, 0)
heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-8)

# Convert grayscale image to RGB
img_rgb = np.stack([img, img, img], axis=-1)

# Convert heatmap to RGB using a colormap
cmap = mpl.colormaps["jet"]
heatmap_rgb = cmap(heatmap_resized)[..., :3]

# Blend image and heatmap
alpha = 0.35
overlay = (1 - alpha) * img_rgb + alpha * heatmap_rgb
overlay = np.clip(overlay, 0, 1)

# Save clean Grad-CAM overlay
plt.figure(figsize=(7, 7))
plt.imshow(overlay)
plt.title(
    f"True: {true_class} | Pred: {pred_class} | P(malignant): {prob:.4f}"
)
plt.axis("off")
plt.savefig("gradcam_overlay.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 7))
plt.imshow(heatmap_resized, cmap="jet")
plt.title(
    f"Grad-CAM heatmap | True: {true_class} | Pred: {pred_class}"
)
plt.axis("off")
plt.savefig("gradcam_heatmap_only.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 7))
plt.imshow(img, cmap="gray")
plt.title(
    f"Original | True: {true_class} | Pred: {pred_class} | P(malignant): {prob:.4f}"
)
plt.axis("off")
plt.savefig("gradcam_original.png", dpi=300, bbox_inches="tight")
plt.close()