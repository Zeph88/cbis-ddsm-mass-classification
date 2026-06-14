# https://keras.io/examples/vision/grad_cam/
# Adaptation of a code from Keras.io, author fchollet

import os
from src.preprocessing.dicom_io import dicom_to_tf_tensor, apply_roi_mask, apply_roi_emphasis, apply_roi_soft_mask
from src.preprocessing.dataset_preprocessing import tensor_to_2d_np, orient_by_breast_mass, crop_zoom_to_roi
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display 
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

# mmg_path="/home/julien/cbis-ddsm/data/preprocessed/soft0.7_full/train_00000.npy"
mmg_path="/home/julien/cbis-ddsm/data/preprocessed/soft0.3_full/train_00010.npy"
array_npy = np.load(mmg_path)
array_npy = np.expand_dims(array_npy, axis=0)

model = tf.keras.models.load_model("model_three_nodes_baseline.keras")
_ = model(array_npy)

last_conv_layer_name = "conv2d_2"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    inputs = model.inputs[0]
    x = inputs

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    predictions = x
    grad_model = keras.models.Model(
        inputs=inputs,
        outputs=[conv_output, predictions]
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


# Print what the top predicted class is
preds = model.predict(array_npy)
print(preds)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(array_npy, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.axis("off")
plt.savefig("gradcam_output.png", dpi=300, bbox_inches="tight")
plt.close()

array_npy = np.squeeze(array_npy)
print(array_npy.shape)
heatmap.resize((224, 224))
print(heatmap.shape)

treated_image = apply_roi_soft_mask(array_npy, heatmap, 0.3)
# Display heatmap
plt.matshow(treated_image)
plt.axis("off")
plt.savefig("gradcam_output_superposed.png", dpi=300, bbox_inches="tight")
plt.close()
