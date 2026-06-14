import numpy as np
import pydicom
import tensorflow as tf


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


def dicom_to_tf_tensor(dicom_path, size=(224, 224)):
    img = read_dicom_as_array(dicom_path)

    return img

def resize_tensor(img, size=(224, 224)):

    img = crop_image(img, 0.15, 0.02, 0.15, 0)

    # [H, W] -> [H, W, 1]
    img = np.expand_dims(img, axis=-1)

    # TensorFlow tensor
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # resize
    tensor = tf.image.resize(tensor, size)

    return tensor