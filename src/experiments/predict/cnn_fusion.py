from src.experiments.models.fusion_global_local import fusion_models
from src.config import OUTPUT_NPY, OUTPUT_MODEL
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.training.dataset_preparation import train_val_test_sets
from src.functions import ensure_directory
import numpy as np
import tensorflow as tf

ensure_directory(OUTPUT_MODEL)

param_inputs = {
    'zoom':{
        'width':256,
        'height':256
    },
    'full':{
        'width':512,
        'height':768
    }
}

local_df = pd.read_csv(OUTPUT_NPY / f"dataset_index_zoom_{param_inputs['zoom']['height']}x{param_inputs['zoom']['width']}.csv")
global_df = pd.read_csv(OUTPUT_NPY / f"dataset_index_full_{param_inputs['full']['height']}x{param_inputs['full']['width']}.csv")

common_id = set(local_df['lesion_key']) & set(global_df['lesion_key'])
local_df = local_df[local_df['lesion_key'].isin(common_id)]
global_df = global_df[global_df['lesion_key'].isin(common_id)]
local_df['local_path'] = local_df['preprocessed_image_path']
global_df['global_path'] = global_df['preprocessed_image_path']
df = pd.merge(local_df, global_df[['global_path', 'lesion_key']], on='lesion_key')
print(df)

train_df, val_df, test_df = train_val_test_sets(df, path_image='local_path', added_path_image='global_path', image_height=param_inputs['zoom']['height'], 
    image_width=param_inputs['zoom']['width'], added_image_height=param_inputs['full']['height'], added_image_width=param_inputs['full']['width'])


local_model = tf.keras.models.load_model(OUTPUT_MODEL / "model_local_branch.keras")
local_model.summary()
global_model = tf.keras.models.load_model(OUTPUT_MODEL / "model_global_branch.keras")
global_model.summary()
feature_conv_layer_name = "dense"
gmic = fusion_models(local_model, global_model, feature_conv_layer_name)

gmic.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print(train_df.element_spec)

for images, labels in train_df.take(1):
    local_images, global_images = images

    print("Local :", local_images.shape)
    print("Global:", global_images.shape)
    print("Labels:", labels.shape)

    print("Modèle local attendu :", local_model.input_shape)
    print("Modèle global attendu:", global_model.input_shape)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL / "model_fusion.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    ),
]

history = gmic.fit(
    train_df,
    validation_data=val_df,
    epochs=100,
    callbacks=callbacks
)

y_true = []
y_prob = []

for images, labels in test_df:
    probs = gmic(
        images,
        training=False,
    ).numpy().ravel()

    y_prob.extend(probs)
    y_true.extend(labels.numpy())

y_true = np.asarray(y_true)
y_prob = np.asarray(y_prob)

print(
    f"minimum probability: {y_prob.min()}, "
    f"maximum probability: {y_prob.max()}, "
    f"average probability: {y_prob.mean()}"
)

auc = roc_auc_score(y_true, y_prob)
print(f"AUC: {auc}")

for threshold in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:

    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    accuracy = (tp + tn) / len(y_true)

    print(f"threshold : {threshold}, accuracy : {accuracy}, precision : {precision}, recall : {recall}")
