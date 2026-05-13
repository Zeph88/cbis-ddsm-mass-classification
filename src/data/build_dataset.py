import os
import pathlib
from pathlib import Path
import pandas as pd 
import numpy as np 
from src.functions import fetch_image, find_dicom_by_series_description
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT, DATASET_INDEX

_MAPPING_PATHOLOGY = {
    'BENIGN' : 0,
    'BENIGN_WITHOUT_CALLBACK' : 0,
    'MALIGNANT' : 1
}

def load_metadata():
    df_train = pd.read_csv(TRAIN_CSV)
    df_train["source"] = "train"

    df_test = pd.read_csv(TEST_CSV)
    df_test["source"] = "test"

    return pd.concat([df_train, df_test], ignore_index=True)

def add_labels(df):
    df = df.copy()
    df["label"] = df["pathology"].map(_MAPPING_PATHOLOGY)

    return df

def add_lesion_key(df):
    df = df.copy()
    df["lesion_key"] = (
        df["patient_id"].astype(str) + "_" +
        df["left or right breast"].astype(str) + "_" +
        df["abnormality id"].astype(str)
    )

    return df

def build_dataset_index():
    df = load_metadata()
    df = add_labels(df)
    df = add_lesion_key(df)

    df["resolved_roi_rel_path"] = df["ROI mask file path"].apply(
        lambda p: find_dicom_by_series_description(
            p,
            IMAGES_ROOT,
            "ROI mask images"
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

    DATASET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_INDEX, index=False)

    print("Saved:", DATASET_INDEX)
    print(df["keep"].value_counts())
    print(f"label 1: {len(df[df["label"]==1])/len(df)}")
    print(f"label 0: {len(df[df["label"]==0])/len(df)}")

if __name__ == "__main__":
    main()