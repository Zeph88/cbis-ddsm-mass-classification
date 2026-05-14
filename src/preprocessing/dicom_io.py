import numpy as np
import pydicom
import tensorflow as tf


def apply_roi_soft_mask(image, mask, background_factor=0.3):
    mask = tf.cast(mask > 0, tf.float32)
    return image * mask + image * (1 - mask) * background_factor

def apply_roi_emphasis(image, mask, alpha=1.5):
    mask = tf.cast(mask > 0, tf.float32)
    emphasized = image * (1 + alpha * mask)
    emphasized = emphasized / tf.reduce_max(emphasized)

    return emphasized

def apply_roi_mask(image, mask):
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


def dicom_to_tf_tensor(dicom_path, size=(224, 224)):
    img = read_dicom_as_array(dicom_path)

    # [H, W] -> [H, W, 1]
    img = np.expand_dims(img, axis=-1)

    # TensorFlow tensor
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # resize
    tensor = tf.image.resize(tensor, size)

    return tensor