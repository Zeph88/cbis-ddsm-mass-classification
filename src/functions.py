import os
from pathlib import Path
import numpy as np
from src.config import SEED
import random
import tensorflow as tf

def set_seed(seed=SEED):
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