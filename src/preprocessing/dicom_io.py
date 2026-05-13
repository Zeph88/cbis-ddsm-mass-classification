import numpy as np
import pydicom
import tensorflow as tf


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