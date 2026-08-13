import os
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from sklearn.metrics import roc_auc_score
from pydicom.data import get_testdata_files
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT, BATCH_SIZE, SEED
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

def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path