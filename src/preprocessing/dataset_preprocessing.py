import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from typing import Union

from src.preprocessing.dicom_io import dicom_to_tf_tensor, apply_roi_mask, apply_roi_emphasis, apply_roi_soft_mask
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY
import matplotlib.pyplot as plt

def save_crop_mask_result(image, mask, masked_image, output_path, label=None, title=None):
    image_np = image.numpy().squeeze()
    mask_np = mask.numpy().squeeze()
    masked_np = masked_image.numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Crop original")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("ROI mask")
    axes[1].axis("off")

    axes[2].imshow(masked_np, cmap="gray")
    axes[2].set_title("Masked result")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title)
    elif label is not None:
        fig.suptitle(f"Label: {label}")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clear_directory(directory_path: Union[str, Path]) -> list:
    """Irreversibly removes all files and folders inside the specified
    directory. Returns a list with paths Python lacks permission to delete."""
    erroneous_paths = []
    for path_object in Path(directory_path).iterdir():
        try:
            if path_object.is_dir():
                shutil.rmtree(path_object)
            else:
                path_object.unlink()
        except PermissionError:
            erroneous_paths.append(path_object)
    return erroneous_paths

def mask_preprocess_roi_images(index_csv=DATASET_INDEX, images_root=IMAGES_ROOT, extended_path="emphasized_mask", mask_function=apply_roi_emphasis):

    clear_directory(OUTPUT_NPY / extended_path)

    debug_dir = OUTPUT_NPY / extended_path / "debug_preview"
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)

    df = df[(df["keep"] == True)].reset_index(drop=True)

    output_paths = []

    OUTPUT_NPY.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        crop_path = images_root / row["resolved_crop_rel_path"]
        roi_path = images_root / row["resolved_roi_rel_path"]

        image = dicom_to_tf_tensor(crop_path)
        mask = dicom_to_tf_tensor(roi_path)

        masked_image = mask_function(image, mask)

        if i < 20:
            save_crop_mask_result(
                image=image,
                mask=mask,
                masked_image=masked_image,
                output_path=debug_dir / f"debug_{i:05d}.png",
                label=row["label"],
                title=f"{row['source']} | index={i} | label={row['label']}"
            )

        output_path = OUTPUT_NPY / extended_path / f"{row['source']}_{i:05d}.npy"

        np.save(output_path, masked_image.numpy().astype("float32"))

        output_paths.append(str(output_path))

    df["preprocessed_roi_path"] = output_paths

    output_csv = OUTPUT_NPY / extended_path / "dataset_index_roi_masked.csv"
    df.to_csv(output_csv, index=False)

    print(f"Saved preprocessed index to: {output_csv}")

if __name__ == "__main__":
    mask_preprocess_roi_images(extended_path="soft_mask", mask_function=apply_roi_soft_mask)