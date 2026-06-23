import os
import pathlib
from pathlib import Path
import pandas as pd 
import numpy as np 
from src.functions import set_seed, check_path
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT, SEED, PROCESSED_DIR
import pydicom

set_seed(SEED)

_MAPPING_PATHOLOGY = {
    'BENIGN' : 0,
    'BENIGN_WITHOUT_CALLBACK' : 0,
    'MALIGNANT' : 1
}

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
            series_description = ds.SeriesDescription

            if series_description == expected_description:
                return str(dcm_abs_path.relative_to(images_root))

        except Exception:
            continue

    return None

def load_metadata():
    df_train = pd.read_csv(TRAIN_CSV)
    df_train = df_train[:10]
    df_train["source"] = "train"

    df_test = pd.read_csv(TEST_CSV)
    df_test = df_test[:5]
    df_test["source"] = "test"

    return pd.concat([df_train, df_test], ignore_index=True)

def add_labels(df):
    df = df.copy()
    df["label"] = df["pathology"].map(_MAPPING_PATHOLOGY)

    return df

def add_lesion_key(df):
    df = df.copy()
    df["lesion_key"] = (
        df["source"].astype(str) + "_" +
        df["patient_id"].astype(str) + "_" +
        df["left or right breast"].astype(str) + "_" +
        df["image view"].astype(str) + "_" +
        df["abnormality id"].astype(str)
    )
    return df

def build_dataset_index():
    print("load_metadata")
    df = load_metadata()
    print("add_labels")
    df = add_labels(df)
    print("add_lesion_key")
    df = add_lesion_key(df)

    print(df["ROI mask file path"][0])
    df["resolved_roi_rel_path"] = df["ROI mask file path"].apply(
        lambda p: find_dicom_by_series_description(
            p,
            IMAGES_ROOT,
            "ROI mask images"
        )
    )
    print(df["resolved_roi_rel_path"][0])

    df["resolved_image_file_path"] = df["image file path"].apply(
        lambda p: find_dicom_by_series_description(
            p,
            IMAGES_ROOT,
            "full mammogram images"
        )
    )

    df["resolved_crop_rel_path"] = df["cropped image file path"].apply(
        lambda p: find_dicom_by_series_description(
            p,
            IMAGES_ROOT,
            "cropped images"
        )
    )

    df["keep"] = df["label"].notna()

    return df


def main():
    df = build_dataset_index()

    DATASET_INDEX = PROCESSED_DIR / "mass_dataset_index_test.csv"

    DATASET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_INDEX, index=False)

    print("Saved:", DATASET_INDEX)
    print(df["keep"].value_counts())
    print(f"label 1: {len(df[df["label"]==1])/len(df)}")
    print(f"label 0: {len(df[df["label"]==0])/len(df)}")

if __name__ == "__main__":
    main()