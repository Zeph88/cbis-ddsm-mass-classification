import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Union, Callable

import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocessing.dicom_io import dicom_to_tf_tensor, apply_roi_mask, apply_roi_emphasis, apply_roi_soft_mask
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY


def clear_directory(directory_path: Union[str, Path]) -> list:
    """
    Irreversibly removes all files and folders inside the specified directory.
    Returns a list with paths Python lacks permission to delete.
    """
    directory_path = Path(directory_path)
    erroneous_paths = []

    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        return erroneous_paths

    for path_object in directory_path.iterdir():
        try:
            if path_object.is_dir():
                shutil.rmtree(path_object)
            else:
                path_object.unlink()
        except PermissionError:
            erroneous_paths.append(path_object)

    return erroneous_paths


def tensor_to_2d_np(x) -> np.ndarray:
    """
    Converts a TensorFlow tensor or NumPy array to a 2D NumPy array.
    Expected input is usually H x W x 1.
    """
    if isinstance(x, tf.Tensor):
        x = x.numpy()

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    elif x.ndim == 3:
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {x.shape}")

    return x


def get_mask_diagnostics(mask_np: np.ndarray) -> dict:
    """
    Returns basic diagnostics about the binary ROI mask.
    """
    mask_bin = mask_np > 0

    h, w = mask_bin.shape
    area = int(mask_bin.sum())
    image_area = int(h * w)
    area_ratio = area / image_area if image_area > 0 else 0.0

    if area == 0:
        return {
            "mask_area": 0,
            "mask_area_ratio": 0.0,
            "bbox_ymin": None,
            "bbox_ymax": None,
            "bbox_xmin": None,
            "bbox_xmax": None,
            "bbox_height": 0,
            "bbox_width": 0,
            "bbox_center_y": None,
            "bbox_center_x": None,
        }

    ys, xs = np.where(mask_bin)

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1

    return {
        "mask_area": area,
        "mask_area_ratio": area_ratio,
        "bbox_ymin": y_min,
        "bbox_ymax": y_max,
        "bbox_xmin": x_min,
        "bbox_xmax": x_max,
        "bbox_height": bbox_height,
        "bbox_width": bbox_width,
        "bbox_center_y": float((y_min + y_max) / 2),
        "bbox_center_x": float((x_min + x_max) / 2),
    }

def orient_by_breast_mass(image, mask=None, target_side="right", threshold=0.05):
    """
    Standardizes visual orientation based on the side where most breast tissue appears.

    Parameters
    ----------
    image:
        TensorFlow tensor, shape (H, W, 1), values expected in [0, 1].
    mask:
        Optional TensorFlow tensor, shape (H, W, 1). If provided, it is flipped
        together with the image.
    target_side:
        "right" means the breast tissue should appear mostly on the right side.
        "left" means the breast tissue should appear mostly on the left side.
    threshold:
        Pixel intensity threshold used to separate tissue from black background.

    Returns
    -------
    image or (image, mask), possibly flipped.
    """

    image_np = image.numpy() if isinstance(image, tf.Tensor) else np.asarray(image)

    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_2d = image_np[:, :, 0]
    else:
        image_2d = np.squeeze(image_np)

    h, w = image_2d.shape

    tissue = image_2d > threshold

    left_mass = tissue[:, : w // 2].sum()
    right_mass = tissue[:, w // 2 :].sum()

    breast_is_left = left_mass > right_mass

    if target_side == "right":
        should_flip = breast_is_left
    elif target_side == "left":
        should_flip = not breast_is_left
    else:
        raise ValueError("target_side must be 'left' or 'right'")

    if should_flip:
        image = tf.image.flip_left_right(image)

        if mask is not None:
            mask = tf.image.flip_left_right(mask)

    if mask is not None:
        return image, mask

    return image

def crop_zoom_to_roi(
    image,
    mask,
    diagnostics,
    output_size=(224, 224),
    margin=30,
):
    """
    Crops the image around the ROI bounding box, optionally applies masking,
    then resizes the crop to output_size.

    mask_mode:
        "none" -> preserve local context
        "hard" -> set pixels outside ROI to 0
        "soft" -> attenuate pixels outside ROI with context_weight
    """

    image_np = tensor_to_2d_np(image).astype("float32")
    mask_np = tensor_to_2d_np(mask).astype("float32")

    if image_np.shape != mask_np.shape:
        raise ValueError(
            f"Image and mask shapes differ: image={image_np.shape}, mask={mask_np.shape}"
        )

    if diagnostics["mask_area"] == 0:
        image_3d = image_np[..., np.newaxis]
        return tf.image.resize(image_3d, output_size)

    y_min = diagnostics["bbox_ymin"]
    y_max = diagnostics["bbox_ymax"]
    x_min = diagnostics["bbox_xmin"]
    x_max = diagnostics["bbox_xmax"]

    h, w = image_np.shape

    y0 = max(0, y_min - margin)
    y1 = min(h, y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(w, x_max + margin + 1)

    crop = image_np[y0:y1, x0:x1]

    if crop.size == 0:
        raise ValueError(
            f"Empty ROI crop: y0={y0}, y1={y1}, x0={x0}, x1={x1}, image_shape={image_np.shape}"
        )

    crop = crop[..., np.newaxis]
    crop = tf.convert_to_tensor(crop, dtype=tf.float32)
    crop = tf.image.resize(crop, output_size)

    return crop

def save_crop_mask_debug(
    image,
    mask,
    masked_image,
    output_path: Path,
    label=None,
    title=None,
):
    """
    Saves a 2x3 debug figure:
    - original mammogram
    - raw mask
    - masked/preprocessed result
    - overlay mask on mammogram
    - contour mask on mammogram
    - zoom around ROI bbox
    """
    image_np = tensor_to_2d_np(image)
    mask_np = tensor_to_2d_np(mask)
    masked_np = tensor_to_2d_np(masked_image)

    if image_np.shape != mask_np.shape:
        raise ValueError(
            f"Image and mask shapes differ: image={image_np.shape}, mask={mask_np.shape}"
        )

    diagnostics = get_mask_diagnostics(mask_np)
    mask_bin = mask_np > 0

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(image_np, cmap="gray")
    axes[0, 0].set_title("Original mammogram")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_np, cmap="gray")
    axes[0, 1].set_title("ROI mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(masked_np, cmap="gray")
    axes[0, 2].set_title("Masked / preprocessed result")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(image_np, cmap="gray")
    axes[1, 0].imshow(mask_bin, cmap="Reds", alpha=0.35)
    axes[1, 0].set_title("Mask overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(image_np, cmap="gray")
    if diagnostics["mask_area"] > 0:
        axes[1, 1].contour(mask_bin, levels=[0.5], colors="red", linewidths=1)
    axes[1, 1].set_title("Mask contour on mammogram")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(image_np, cmap="gray")
    if diagnostics["mask_area"] > 0:
        y_min = diagnostics["bbox_ymin"]
        y_max = diagnostics["bbox_ymax"]
        x_min = diagnostics["bbox_xmin"]
        x_max = diagnostics["bbox_xmax"]

        margin = 30
        h, w = image_np.shape

        y0 = max(0, y_min - margin)
        y1 = min(h, y_max + margin)
        x0 = max(0, x_min - margin)
        x1 = min(w, x_max + margin)

        axes[1, 2].set_xlim(x0, x1)
        axes[1, 2].set_ylim(y1, y0)
        axes[1, 2].contour(mask_bin, levels=[0.5], colors="red", linewidths=1)

    axes[1, 2].set_title("Zoom around ROI")
    axes[1, 2].axis("off")

    info = (
        f"mask area={diagnostics['mask_area']} | "
        f"ratio={diagnostics['mask_area_ratio']:.4%} | "
        f"bbox={diagnostics['bbox_width']}x{diagnostics['bbox_height']}"
    )

    if title is not None:
        fig.suptitle(f"{title}\n{info}")
    elif label is not None:
        fig.suptitle(f"Label: {label}\n{info}")
    else:
        fig.suptitle(info)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return diagnostics


def mask_preprocess_roi_images(
    df,
    images_root=IMAGES_ROOT,
    debug_limit: int = 50,
    zoom_to_roi: bool = False,
    zoom_margin: int = 30,
    mask_mode: str = "none",
    factor: float = 0.25
):
    
    if zoom_to_roi:
        zoom_path = "zoom" + str(zoom_margin)
    else:
        zoom_path = "full"
    
    if mask_mode == "soft":
        mask_function = apply_roi_soft_mask
        mask_path = mask_mode + str(factor)
    elif mask_mode == "emphasis":
        mask_function = apply_roi_emphasis
        mask_path = mask_mode + str(factor)
    elif mask_mode == "hard":
        mask_function = apply_roi_mask
        mask_path = mask_mode
    else:
        mask_function = None
        mask_path = "nomask"

    extended_path = f"{mask_path}_{zoom_path}"

    output_dir = OUTPUT_NPY / extended_path
    output_dir_exists = os.path.exists(output_dir)

    if output_dir_exists:
        clear_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_preview"
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["keep"] == True].reset_index(drop=True)

    output_paths = []
    diagnostics_rows = []

    OUTPUT_NPY.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        mmg_path = images_root / row["resolved_image_file_path"]
        roi_path = images_root / row["resolved_roi_rel_path"]

        image = dicom_to_tf_tensor(dicom_path=mmg_path)
        mask = dicom_to_tf_tensor(dicom_path=roi_path)

        image, mask = orient_by_breast_mass(image, mask)

        image_np = tensor_to_2d_np(image)
        mask_np = tensor_to_2d_np(mask)

        if image_np.shape != mask_np.shape:
            raise ValueError(
                f"Shape mismatch at index={i}: "
                f"mammogram={image_np.shape}, mask={mask_np.shape}, "
                f"mmg_path={mmg_path}, roi_path={roi_path}"
            )

        diagnostics = get_mask_diagnostics(mask_np)

        if mask_function != None:
            treated_image = mask_function(image, mask, factor)
        else:
            treated_image = image

        if zoom_to_roi:
            treated_image = crop_zoom_to_roi(
                image=treated_image,
                mask=mask,
                diagnostics=diagnostics,
                output_size=(224, 224),
                margin=zoom_margin,
            )    

        diagnostics_row = {
            "index": i,
            "sample_id": row.get("sample_id", None), 
            "source": row.get("source", None),
            "label": row.get("label", None),
            "mmg_path": str(mmg_path),
            "roi_path": str(roi_path),
            "image_height": image_np.shape[0],
            "image_width": image_np.shape[1],
            **diagnostics,
        }

        diagnostics_rows.append(diagnostics_row)

        if i < debug_limit:
            save_crop_mask_debug(
                image=image,
                mask=mask,
                masked_image=treated_image,
                output_path=debug_dir / f"debug_{i:05d}.png",
                label=row["label"],
                title=f"{row['source']} | index={i} | label={row['label']}",
            )

        output_path = output_dir / f"{row['source']}_{i:05d}.npy"

        np.save(output_path, treated_image.numpy().astype("float32"))
        output_paths.append(str(output_path))

    df["preprocessed_image_path"] = output_paths

    output_csv = OUTPUT_NPY / f"dataset_index_{extended_path}.csv"
    df.to_csv(output_csv, index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_csv = output_dir / "mask_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    print(f"Saved preprocessed index to: {output_csv}")
    print(f"Saved mask diagnostics to: {diagnostics_csv}")
    print(f"Saved debug previews to: {debug_dir}")

def crop_save_vanilla(df):
    output_dir = OUTPUT_NPY / "cropped"
    clear_directory(output_dir)

    df = df[df["keep"] == True].reset_index(drop=True)

    output_paths = []

    OUTPUT_NPY.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        crop_path = IMAGES_ROOT / row["resolved_crop_rel_path"]

        image = dicom_to_tf_tensor(dicom_path=crop_path)

        if isinstance(image, tf.Tensor):
            image_np = image.numpy()
        else:
            image_np = np.asarray(image)

        # Ensure shape is H, W, 1
        if image_np.ndim == 2:
            image_np = image_np[..., np.newaxis]

        if image_np.ndim != 3 or image_np.shape[-1] != 1:
            raise ValueError(
                f"Expected crop image shape (H, W, 1), got {image_np.shape} "
                f"for crop_path={crop_path}"
            )

        output_path = output_dir / f"{row['source']}_{i:05d}.npy"

        np.save(output_path, image_np.astype("float32"))
        output_paths.append(str(output_path))

    df["preprocessed_image_path"] = output_paths

    output_csv = OUTPUT_NPY / "dataset_index_cropped.csv"
    df.to_csv(output_csv, index=False)

    print(f"Saved preprocessed index to: {output_csv}")

def add_sample_id(df):
    df = df.copy()

    if "sample_id" not in df.columns:
        df["sample_id"] = (
            df["source"].astype(str) + "_" +
            df["patient_id"].astype(str) + "_" +
            df["left or right breast"].astype(str) + "_" +
            df["image view"].astype(str) + "_" +
            df["abnormality id"].astype(str)
        )

    df["sample_id"] = (
        df["sample_id"]
        .astype(str)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df

if __name__ == "__main__":

    train_df = pd.read_csv(OUTPUT_NPY / "train_split.csv")
    val_df = pd.read_csv(OUTPUT_NPY / "val_split.csv")
    test_df = pd.read_csv(OUTPUT_NPY / "test_split.csv")

    train_df["set"] = "train"
    val_df["set"] = "validation"
    test_df["set"] = "test"

    df = pd.concat(
        [train_df, val_df, test_df],
        axis=0,
        ignore_index=True
    )

    df = add_sample_id(df)

    crop_run = False
    mask_run = True

    if mask_run:
         mask_preprocess_roi_images(
            df,
            zoom_to_roi=False,
            #zoom_margin=30,
            mask_mode="soft",
            factor=0.7,
            debug_limit=50,
        )
    
    if crop_run:
        crop_save_vanilla(df)
