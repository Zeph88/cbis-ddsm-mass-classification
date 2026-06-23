import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Union, Callable
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from src.preprocessing.dicom_handling import read_dicom_as_array, crop_image, remove_annotations, fix_border
from src.config import IMAGES_ROOT, OUTPUT_NPY, PIXELS_H, PIXELS_W, CROP_SIZE


import numpy as np
import tensorflow as tf
import cv2


def crop_breast_to_target_ratio(
    image,
    target_size=(768, 512),
    threshold_ratio=0.01,
    margin_ratio=0.02,
):
    """
    Crops tightly around the breast, then adjusts the crop by removing
    excess width or height to match the target aspect ratio.

    No padding and no expansion into background are performed.
    """

    image_np = (
        image.numpy()
        if isinstance(image, tf.Tensor)
        else np.asarray(image)
    )

    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_2d = image_np[:, :, 0]
    elif image_np.ndim == 2:
        image_2d = image_np
    else:
        raise ValueError(
            f"Expected (H, W) or (H, W, 1), got {image_np.shape}"
        )

    if not np.isfinite(image_2d).all():
        raise ValueError("Image contains NaN or infinite values.")

    height, width = image_2d.shape

    image_min = float(image_2d.min())
    image_max = float(image_2d.max())

    if image_max <= image_min:
        raise ValueError("Image has no usable intensity range.")

    threshold = (
        image_min
        + threshold_ratio * (image_max - image_min)
    )

    foreground = (
        image_2d > threshold
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )
    )

    if number_of_labels <= 1:
        raise ValueError("No breast component found.")

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    best_label = 1 + int(
        np.argmax(component_areas)
    )

    x = int(
        stats[best_label, cv2.CC_STAT_LEFT]
    )
    y = int(
        stats[best_label, cv2.CC_STAT_TOP]
    )
    box_width = int(
        stats[best_label, cv2.CC_STAT_WIDTH]
    )
    box_height = int(
        stats[best_label, cv2.CC_STAT_HEIGHT]
    )

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    x_min = max(0, x - margin_x)
    x_max = min(
        width,
        x + box_width + margin_x,
    )

    y_min = max(0, y - margin_y)
    y_max = min(
        height,
        y + box_height + margin_y,
    )

    crop = image_2d[
        y_min:y_max,
        x_min:x_max,
    ]

    crop_height, crop_width = crop.shape

    target_height, target_width = target_size
    target_ratio = target_width / target_height
    current_ratio = crop_width / crop_height

    if current_ratio < target_ratio:
        # Crop too narrow:
        # reduce height rather than expanding into black background.
        desired_height = int(
            round(crop_width / target_ratio)
        )

        desired_height = min(
            desired_height,
            crop_height,
        )

        excess_height = (
            crop_height - desired_height
        )

        crop_top = excess_height // 2
        crop_bottom = (
            excess_height - crop_top
        )

        crop = crop[
            crop_top:
            crop_height - crop_bottom,
            :
        ]

    elif current_ratio > target_ratio:
        # Crop too wide:
        # reduce width while preserving the right side.
        desired_width = int(
            round(crop_height * target_ratio)
        )

        desired_width = min(
            desired_width,
            crop_width,
        )

        # Breast is oriented to the right:
        # remove excess width from the left.
        crop = crop[
            :,
            crop_width - desired_width:,
        ]

    crop = crop[..., np.newaxis]

    resized = tf.image.resize(
        crop,
        size=target_size,
        method="bilinear",
        antialias=True,
    )

    resized = tf.clip_by_value(
        resized,
        0.0,
        1.0,
    )

    resized.set_shape(
        (
            target_height,
            target_width,
            1,
        )
    )

    return resized

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

    image_np = tensor_to_2d_np(image).astype("float32")
    mask_np = tensor_to_2d_np(mask).astype("float32")

    if image_np.shape != mask_np.shape:
        raise ValueError(
            f"Image and mask shapes differ: image={image_np.shape}, mask={mask_np.shape}"
        )

    crop  = geometric_center(image_np, mask_np)

    crop = crop[..., np.newaxis]
    crop = tf.convert_to_tensor(crop, dtype=tf.float32)
    
    if crop.shape[:2] != output_size:
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
    debug_limit: int = 5,
    zoom_to_roi: bool = False,
    resolution = (598, 598)
):
    
    if zoom_to_roi:
        zoom_path = f"zoom_{resolution[0]}x{resolution[1]}_test"
    else:
        zoom_path = f"full_{resolution[0]}x{resolution[1]}_test"

    output_dir = OUTPUT_NPY / zoom_path
    output_dir_exists = os.path.exists(output_dir)

    if output_dir_exists:
        clear_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_preview"
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["keep"] == True].reset_index(drop=True)

    processed_rows = []
    output_paths = []

    OUTPUT_NPY.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_shape_mismatch = 0
    for i, row in df.iterrows():
        mmg_path = images_root / row["resolved_image_file_path"]
        roi_path = images_root / row["resolved_roi_rel_path"]

        image = read_dicom_as_array(dicom_path=mmg_path)
        mask = read_dicom_as_array(dicom_path=roi_path)

        start = time.perf_counter()
        image, mask = orient_by_breast_mass(image, mask)
        print(f"orient_image_to_right: {time.perf_counter() - start:.3f}s")

        image_np = tensor_to_2d_np(image)
        mask_np = tensor_to_2d_np(mask)
        
        start = time.perf_counter()
        # Control the size of the image matches the size of the mask.
        if image_np.shape != mask_np.shape:
            print(
                f"Shape mismatch at index={i}: "
                f"mammogram={image_np.shape}, mask={mask_np.shape}, "
                f"mmg_path={mmg_path}, roi_path={roi_path}"
            )
            skipped_shape_mismatch += 1
            continue
        print(f"control1: {time.perf_counter() - start:.3f}s")

        start = time.perf_counter()
        # Control the mask displays a ROI
        if np.count_nonzero(mask_np > 0) == 0:
            print(
                f"The mask contains no positive pixel at index={i}: "
                f"mmg_path={mmg_path}, roi_path={roi_path}"
            )
            skipped_shape_mismatch += 1
            continue
        print(f"control2: {time.perf_counter() - start:.3f}s")

        if zoom_to_roi:
            treated_image = crop_zoom_to_roi(
                image=image,
                mask=mask,
                output_size=resolution  
            ) 
        else:
            
            start = time.perf_counter()
            breast_crop = remove_annotations(image)
            print(f"remove_annotations: {time.perf_counter() - start:.3f}s")
            treated_image = crop_breast_to_target_ratio(
                breast_crop,
                target_size=(PIXELS_H, PIXELS_W),
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

        start = time.perf_counter()
        np.save(output_path, treated_image)
        print(f"save_numpy: {time.perf_counter() - start:.3f}s")
        processed_rows.append(row.to_dict())
        output_paths.append(str(output_path))

    processed_df = pd.DataFrame(processed_rows)
    processed_df["preprocessed_image_path"] = output_paths

    output_csv = OUTPUT_NPY / f"dataset_index_{zoom_path}.csv"
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

    train_df = pd.read_csv(OUTPUT_NPY / "train_split_test.csv")
    test_df = pd.read_csv(OUTPUT_NPY / "train_split_test.csv")

    train_df["set"] = "train"
    test_df["set"] = "test"

    df = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True
    )

    print(f"{len(df)} lines")

    df = add_sample_id(df)

    crop_run = False

    preprocess_images(df, zoom_to_roi=crop_run, resolution=(PIXELS_H, PIXELS_W))
    
