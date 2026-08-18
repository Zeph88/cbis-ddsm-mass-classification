import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Union, Callable
import time
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocessing.dicom_handling import read_dicom_as_array, crop_image, resize_with_padding, remove_annotations, fix_border, tensor_to_2d_np, crop_breast_to_target_ratio
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY, SPLITS_DIR, LOCAL_HEIGHT, LOCAL_WIDTH, GLOBAL_HEIGHT, GLOBAL_WIDTH, CROP_SIZE
from src.functions import load_data


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

def flip_left_right_any(x):
    if isinstance(x, tf.Tensor):
        if len(x.shape) == 2:
            return tf.reverse(x, axis=[1])
        return tf.image.flip_left_right(x)
    else:
        x = np.asarray(x)
        return np.fliplr(x)

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
        image = flip_left_right_any(image)

        if mask is not None:
            mask = flip_left_right_any(mask)

    if mask is not None:
        return image, mask

    return image


def geometric_center(image, mask):

    height, width = image.shape[:2]

    if height < CROP_SIZE or width < CROP_SIZE:
        raise ValueError(
            f"Image is smaller than the requested crop: "
            f"image_shape={(height, width)}, crop_size={CROP_SIZE}"
        )

    ys, xs = np.nonzero(mask > 0)

    if len(xs) == 0:
        raise ValueError("The mask contains no positive pixels.")

    x_c = float(xs.mean())
    y_c = float(ys.mean())

    half = CROP_SIZE // 2

    x_start = int(round(x_c)) - half
    y_start = int(round(y_c)) - half

    # Shift the complete window inside the image.
    x_start = max(0, min(x_start, width - CROP_SIZE))
    y_start = max(0, min(y_start, height - CROP_SIZE))

    x_end = x_start + CROP_SIZE
    y_end = y_start + CROP_SIZE

    crop = image[y_start:y_end, x_start:x_end]

    return crop

def crop_zoom_to_roi(
    image,
    mask,
    output_size=(598, 598)
):
    """
    Extracts a fixed-size square crop centred on the lesion-mask centroid,
    then resizes the crop to output_size.
    """

    if image.ndim!=2 or mask.ndim!=2:
        raise ValueError("Both mask and mammogram should be in 2D for proper handling")

    crop  = geometric_center(image, mask)

    # crop = crop[..., np.newaxis]
    # crop = tf.convert_to_tensor(crop, dtype=tf.float32)
    
    if crop.shape[:2] != output_size:
        crop = crop[..., np.newaxis]
        crop = tf.image.resize(crop, output_size)
        crop = tf.squeeze(crop, axis=-1)

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
    # image_np = tensor_to_2d_np(image)
    # mask_np = tensor_to_2d_np(mask)
    # masked_np = tensor_to_2d_np(masked_image)
    image_np = image
    mask_np = mask
    masked_np = masked_image

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


def preprocess_images(
    df,
    images_root=IMAGES_ROOT,
    debug_limit: int = 20,
    zoom_to_roi: bool = False,
    resolution = (598, 598)
):
    
    # Define the relative file path and the pipeline depending the mammogram is cropped or not
    if zoom_to_roi:
        file_path = f"zoom_{resolution[0]}x{resolution[1]}"
    else:
        file_path = f"full_{resolution[0]}x{resolution[1]}"

    # Set the absolute file path
    output_dir = OUTPUT_NPY / file_path
    output_dir_exists = os.path.exists(output_dir)

    # If it exists, clean the folder. If not, create it.
    if output_dir_exists:
        clear_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create a folder to save the debug files
    debug_dir = output_dir / "debug_preview"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Filter records to those that have a 0 or 1 label
    df = df[df["keep"] == True].reset_index(drop=True)

    # processed_rows retains all cases that can be used after the preprocessing was successfully performed
    processed_rows = []
    # output_paths adds the location of the processed image and adds a column to processed_rows dataframe 
    output_paths = []


    # OUTPUT_NPY.mkdir(parents=True, exist_ok=True)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # counts the cases that were set out of scope
    skipped_shape_mismatch = 0

    # Loop over all sets
    for i, row in df.iterrows():

        # Retrieve absolute path to DICOMs
        mmg_path = images_root / row["resolved_image_file_path"]
        roi_path = images_root / row["resolved_roi_rel_path"]

        # Convert DICOMs to 2D numpy arrays and normalize values form 0 to 1 
        image = read_dicom_as_array(dicom_path=mmg_path)
        mask = read_dicom_as_array(dicom_path=roi_path)

        # Orient the breast tissue to the right side of the image
        image, mask = orient_by_breast_mass(image, mask)
        

        # image_np = tensor_to_2d_np(image)
        # mask_np = tensor_to_2d_np(mask)
        

        # Control the size of the image matches the size of the mask.
        if image.shape != mask.shape:
            print(
                f"Shape mismatch at index={i}: "
                f"mammogram={image.shape}, mask={mask.shape}, "
                f"mmg_path={mmg_path}, roi_path={roi_path}"
            )
            skipped_shape_mismatch += 1
            continue
        
        # Control the mask displays a ROI
        if np.count_nonzero(mask > 0) == 0:
            print(
                f"The mask contains no positive pixel at index={i}: "
                f"mmg_path={mmg_path}, roi_path={roi_path}"
            )
            skipped_shape_mismatch += 1
            continue
        

        if zoom_to_roi:
            treated_image = crop_zoom_to_roi(
                image=image,
                mask=mask,
                output_size=resolution  
            )
            treated_image = treated_image[..., np.newaxis]
        else:
            
            # Remove annotations from the mammogram
            breast_crop = remove_annotations(image)
            
            # Crop image to remove white borders and excessive padding, which could create noise
            treated_image =  crop_breast_to_target_ratio(
                breast_crop,
                target_size=(resolution[0], resolution[1]),
                threshold_ratio=0.01,
                margin_ratio=0.03,
            )
            
            
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
        processed_rows.append(row.to_dict())
        output_paths.append(str(output_path))

    processed_df = pd.DataFrame(processed_rows)
    processed_df["preprocessed_image_path"] = output_paths

    output_csv = OUTPUT_NPY / f"dataset_index_{file_path}.csv"
    processed_df.to_csv(output_csv, index=False)

    print(f"Saved preprocessed index to: {output_csv}")
    print(f"Saved debug previews to: {debug_dir}")
    print(f"Skipped shape mismatches: {skipped_shape_mismatch}")

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

    parser = argparse.ArgumentParser(
        description="Preprocess CBIS-DDSM images for the local or global branch."
    )

    parser.add_argument(
        "--mode",
        choices=["local", "global"],
        required=True,
        help="Preprocessing mode: 'local' for ROI-centred crops, "
             "'global' for full mammograms."
    )

    args = parser.parse_args()

    if args.mode == "local":
        add_path = ""
        zoom_to_roi = True
        resolution = (LOCAL_HEIGHT, LOCAL_WIDTH)
    else:
        add_path = "_global"
        zoom_to_roi = False
        resolution = (GLOBAL_HEIGHT, GLOBAL_WIDTH)

    train_df, val_df, test_df = load_data(SPLITS_DIR / f"train_split{add_path}.csv", SPLITS_DIR / f"val_split{add_path}.csv", SPLITS_DIR / f"test_split{add_path}.csv")

    train_df["set"] = "train"
    val_df["set"] = "validation"
    test_df["set"] = "test"

    df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    df = add_sample_id(df)

    print(
        f"Running {args.mode} preprocessing "
        f"with resolution {resolution}"
    )

    preprocess_images(df, zoom_to_roi=zoom_to_roi, resolution=resolution)
    
