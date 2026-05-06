from pathlib import Path
import pandas as pd
import numpy as np
import pydicom

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

    if path_folder.exists():
        real_folder = True
    else:
        real_folder = False

    return real_folder, path_folder
