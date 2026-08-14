import os
from src.preprocessing.dataset_preprocessing import tensor_to_2d_np, orient_by_breast_mass, crop_zoom_to_roi
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def fusion_models(local_model, global_model, feature_conv_layer_name):

    regularizer = tf.keras.regularizers.l2(1e-4)

    local_extractor = tf.keras.Model(
        inputs=local_model.input,
        outputs=local_model.get_layer(feature_conv_layer_name).output,
        name="local_feature_extractor",
    )

    global_extractor = tf.keras.Model(
        inputs=global_model.input,
        outputs=global_model.get_layer(feature_conv_layer_name).output,
        name="global_feature_extractor",
    )

    local_extractor.trainable = False
    global_extractor.trainable = False

    local_input = tf.keras.Input(shape=local_model.input_shape[1:], name="local_input")
    global_input = tf.keras.Input(shape=global_model.input_shape[1:], name="global_input")

    local_x = local_extractor(local_input, training=False)
    global_x = global_extractor(global_input, training=False)
    
    # ====================================================================================
    # Regularization (2nd improvement)
    # ====================================================================================

    local_x = tf.keras.layers.LayerNormalization(name="local_normalization")(local_x)
    global_x = tf.keras.layers.LayerNormalization(name="global_normalization")(global_x)

    local_x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer, name="local_projection")(local_x)
    global_x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer, name="global_projection")(global_x)
    
    # ====================================================================================
    # Regularization (2nd improvement)
    # ====================================================================================

    x = tf.keras.layers.Concatenate(name="feature_fusion")([
        global_x,
        local_x
    ])

    # x = tf.keras.layers.Dense(128, activation="relu", name="fusion_dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="fusion_dropout")(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fusion_output"
    )(x)

    model = tf.keras.Model(
        inputs=[
            local_input,
            global_input
        ],
        outputs=outputs,
        name="global_local_fusion"
    )

    return model

