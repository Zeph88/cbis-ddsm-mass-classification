import os
from pathlib import Path
import numpy as np
from src.config import SEED, SPLITS_DIR
import random
import tensorflow as tf
import pandas as pd
import argparse

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

def load_data(*paths):
    
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        files.append(pd.read_csv(path))

    return files

def parse_arguments(
    description,
    arguments=None,
    exclusive_arguments=None,
    exclusive_required=False,
):
    parser = argparse.ArgumentParser(
        description=description
    )

    if arguments:
        for argument in arguments:
            argument = argument.copy()
            name = argument.pop("name")

            parser.add_argument(
                name,
                **argument,
            )

    if exclusive_arguments:
        group = parser.add_mutually_exclusive_group(
            required=exclusive_required
        )

        for argument in exclusive_arguments:
            argument = argument.copy()
            name = argument.pop("name")

            group.add_argument(
                name,
                **argument,
            )

    return parser.parse_args()
