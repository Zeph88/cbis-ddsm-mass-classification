import numpy as np
import pydicom
import tensorflow as tf
from scipy import ndimage
import cv2
from src.config import PIXELS_H, PIXELS_W

def tensor_to_2d_np(x) -> np.ndarray:

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

def crop_breast_to_target_ratio(
    image,
    target_size=(PIXELS_H, PIXELS_W),
    threshold_ratio=0.01,
    margin_ratio=0.02,
    mask=None
):
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
        raise ValueError(f"Expected (H, W) or (H, W, 1), got {image_np.shape}")

    if not np.isfinite(image_2d).all():
        raise ValueError("Image contains NaN or infinite values.")

    mask_2d = None

    if mask is not None:
        mask_np = (mask.numpy() if isinstance(mask, tf.Tensor) else np.asarray(mask))

        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_2d = mask_np[:, :, 0]
        elif mask_np.ndim == 2:
            mask_2d = mask_np
        
        if mask_2d.shape != image_2d.shape:
            raise ValueError(f"Image and mask shapes differ: image={image_2d.shape}, mask={mask_2d.shape}")

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

    mask_crop = None
    if mask_2d is not None:
        mask_crop = mask_2d[y_min:y_max, x_min:x_max]

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

        if mask_crop is not None:
            mask_crop = mask_crop[crop_top:crop_height - crop_bottom, :]

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

        if mask_crop is not None:
            mask_crop = mask_crop[:, crop_width - desired_width:]

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

    if mask_crop is None:
        return resized

    mask_crop = mask_crop[..., np.newaxis]

    resized_mask = tf.image.resize(mask_crop, size=target_size, method="nearest")
    resized_mask = tf.cast(resized_mask > 0.5, tf.uint8)
    resized_mask.set_shape((target_height, target_width, 1))

    return resized, resized_mask

def apply_roi_soft_mask(image, mask, factor=0.3):
    mask = tf.cast(mask > 0, tf.float32)
    return image * mask + image * (1 - mask) * factor

def apply_roi_emphasis(image, mask, factor=0.5):
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask > 0, tf.float32)

    if len(mask.shape) == 2:
        mask = tf.expand_dims(mask, axis=-1)

    emphasized = image * (1.0 + factor * mask)

    return tf.clip_by_value(emphasized, 0.0, 1.0)

def apply_roi_mask(image, mask, factor=0):
    mask = tf.cast(mask > 0, tf.float32)

    if len(mask.shape) == 2:
        mask = tf.expand_dims(mask, axis=-1)

    return image * mask

def read_dicom_as_array(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    photometric = str(getattr(ds, "PhotometricInterpretation", ""))

    if photometric == "MONOCHROME1":
        img = img.max() + img.min() - img

    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return img


def crop_image(img, top=0.0, bottom=0.0, left=0.0, right=0.0):
    h, w = img.shape[:2]
    top_px = int(h * top)
    bottom_px = int(h * bottom)
    left_px = int(w * left)
    right_px = int(w * right)
    img = img[
        top_px:h - bottom_px,
        left_px:w - right_px
    ]
    return img

def resize_tensor(img, size=(224, 224), zoom_applied=False):

    if zoom_applied==False:
        img = crop_image(img, 0.15, 0.02, 0.15, 0)

    # [H, W] -> [H, W, 1]
    img = np.expand_dims(img, axis=-1)

    # TensorFlow tensor
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # resize
    tensor = tf.image.resize(tensor, size)

    return tensor


def resize_with_padding(
    image,
    target_size=(512, 512),
):
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    if image.shape.rank == 2:
        image = image[..., tf.newaxis]

    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]

    target_height, target_width = target_size

    scale = tf.minimum(
        target_height / tf.cast(original_height, tf.float32),
        target_width / tf.cast(original_width, tf.float32),
    )

    new_height = tf.cast(
        tf.round(tf.cast(original_height, tf.float32) * scale),
        tf.int32,
    )
    new_width = tf.cast(
        tf.round(tf.cast(original_width, tf.float32) * scale),
        tf.int32,
    )

    resized = tf.image.resize(
        image,
        size=(new_height, new_width),
        method="bilinear",
        antialias=True,
    )

    pad_height = target_height - new_height
    pad_width = target_width - new_width

    # Vertical padding: center
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    # Horizontal padding: push all padding to the LEFT
    # so the breast stays aligned to the RIGHT.
    pad_left = pad_width
    pad_right = 0

    padded = tf.pad(
        resized,
        paddings=[
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0],
        ],
        mode="CONSTANT",
        constant_values=0.0,
    )

    return padded


def remove_annotations(
    image: np.ndarray,
    threshold_ratio: float = 0.01,
    min_component_ratio: float = 0.01,
    margin: int = 5,
    orient_right: bool = True,
):

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, received shape {image.shape}."
        )

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    height, width = image.shape

    image_min = float(image.min())
    image_max = float(image.max())

    if image_max <= image_min:
        raise ValueError("Image has no usable intensity range.")

    threshold = image_min + threshold_ratio * (image_max - image_min)

    foreground = image > threshold
    foreground_u8 = foreground.astype(np.uint8) * 255

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    foreground_u8 = cv2.morphologyEx(
        foreground_u8,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    foreground_u8 = cv2.morphologyEx(
        foreground_u8,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            foreground_u8,
            connectivity=8,
        )
    )

    if number_of_labels <= 1:
        raise ValueError("No foreground components found.")

    minimum_area = min_component_ratio * image.size

    best_label = None
    best_score = -np.inf

    for label_id in range(1, number_of_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        component_width = stats[label_id, cv2.CC_STAT_WIDTH]
        component_height = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < minimum_area:
            continue

        area_ratio = area / image.size
        width_ratio = component_width / width
        height_ratio = component_height / height

        x_max = x + component_width - 1

        if orient_right:
            side_contact = x_max / max(width - 1, 1)
        else:
            side_contact = 1.0 - x / max(width - 1, 1)

        score = (
            5.0 * area_ratio
            + 1.5 * height_ratio
            + 1.0 * width_ratio
            + 0.5 * side_contact
        )

        if score > best_score:
            best_score = score
            best_label = label_id

    if best_label is None:
        raise ValueError("No plausible breast component found.")

    cleaned_image = np.zeros_like(image)
    cleaned_image[labels == best_label] = image[labels == best_label]

    return cleaned_image



def fix_border(image,x_pad,y_pad):

    height, width = image.shape

    coef_x = x_pad / width
    coef_y = y_pad // 2 / height

    start_x = coef_x * width
    end_x = width
    start_y = coef_y * height
    end_y = (1-coef_y) * height

    start_x = tf.cast(start_x, tf.int32)
    end_x = tf.cast(end_x, tf.int32)
    start_y = tf.cast(start_y, tf.int32)
    end_y = tf.cast(end_y, tf.int32)

    image = image[
        start_y:end_y,
        start_x:end_x,
        :
    ]

    resized = tf.image.resize(
        image,
        size=(PIXELS_H, PIXELS_W),
        method="bilinear",
        antialias=True,
    )

    return resized


