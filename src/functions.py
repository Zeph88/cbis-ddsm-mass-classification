import os
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from sklearn.metrics import roc_auc_score
from pydicom.data import get_testdata_files
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT, OUTPUT_NPY, BATCH_SIZE, SEED
from src.training.dataset_preparation import build_tf_dataset
import random
import tensorflow as tf
import math

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def fetch_image(path_csv_train, path_csv_test, image_root):
    csv_path_train = Path(path_csv_train)
    csv_path_test = Path(path_csv_test)
    images_root = Path(image_root)

    df_train = pd.read_csv(csv_path_train)
    df_test = pd.read_csv(csv_path_test)

    return df_train, df_test, images_root


def check_path(val, images_root):

    rel_path = list(Path(str(val).strip()).parts)
    rel_path = "/".join(rel_path[:-1])
    path_folder = Path(images_root / rel_path)
    relative_path = Path(rel_path)

    if path_folder.exists():
        real_folder = True
    else:
        real_folder = False

    return real_folder, relative_path

def find_dicom_by_series_description(csv_path_value, images_root, expected_description):
    """
    Resolve one CBIS-DDSM CSV path to the actual DICOM file whose
    SeriesDescription matches expected_description.

    Returns a relative path to images_root, or None.
    """
    isfolder, rel_folder = check_path(csv_path_value, images_root)

    if not isfolder:
        return None

    abs_folder = images_root / rel_folder

    for dcm_abs_path in abs_folder.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(dcm_abs_path, stop_before_pixels=True)
            series_description = str(
                getattr(ds, "SeriesDescription", "")
            ).strip()

            if series_description == expected_description:
                return str(dcm_abs_path.relative_to(images_root))

        except Exception:
            continue

    return None

def train_val_test_sets(
        data,
        batch_size=BATCH_SIZE, 
        seed=SEED
        ):

    train_ds = build_tf_dataset(
        data[data["set"]=="train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_ds = build_tf_dataset(
        data[data["set"]=="validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed
    )

    test_ds = build_tf_dataset(
        data[data["set"]=="test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed
    )

    return train_ds, val_ds, test_ds

def cnn_steps(data):

    train_size = len(data[data["set"]=="train"])
    val_size = len(data[data["set"]=="validation"])
    test_size = len(data[data["set"]=="test"])

    train_steps = math.ceil(train_size / BATCH_SIZE)
    val_steps = math.ceil(val_size / BATCH_SIZE)
    test_steps = math.ceil(test_size / BATCH_SIZE)

    print(f"Train: {train_size} samples, {train_steps} steps")
    print(f"Val: {val_size} samples, {val_steps} steps")
    print(f"Test: {test_size} samples, {test_steps} steps")

    return train_steps, val_steps, test_steps

def cnn_predict(model, test_dataset):
    y_true = []
    y_prob = []

    for images, labels in test_dataset:
        probs = model(images, training=False).numpy().ravel()
        y_prob.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    print(f"minimum probability : {y_prob.min()}, maximum probability : {y_prob.max()}, average probability : {y_prob.mean()}")

    return y_prob, y_true

def evaluate_thresholds(y_prob, y_true, thresholds=[0.35, 0.40, 0.45, 0.50]):
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc}")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        accuracy = (tp + tn) / len(y_true)

        print(f"threshold : {threshold}, accuracy : {accuracy}, precision : {precision}, recall : {recall}")

    