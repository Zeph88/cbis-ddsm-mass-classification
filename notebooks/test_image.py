from pathlib import Path
import argparse

import numpy as np
import pydicom
from PIL import Image


def dicom_to_uint8(ds):
    """
    Converts a DICOM pixel array to visible uint8 PNG.

    Uses:
    - RescaleSlope / RescaleIntercept if present
    - MONOCHROME1 inversion if needed
    - percentile clipping for visibility
    """
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    # MONOCHROME1 means high values are dark, so invert for normal display
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    # If this is a binary ROI mask, keep it binary-looking
    unique_values = np.unique(arr)
    if len(unique_values) <= 5:
        arr = (arr > 0).astype(np.float32) * 255.0
        return arr.astype(np.uint8)

    # Robust contrast scaling
    low, high = np.percentile(arr, [1, 99])

    if high <= low:
        low, high = arr.min(), arr.max()

    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    arr = arr * 255.0

    return arr.astype(np.uint8)


def convert_dicom_file(dcm_path: Path, output_dir: Path):
    ds = pydicom.dcmread(str(dcm_path))

    img_uint8 = dicom_to_uint8(ds)

    series_description = str(getattr(ds, "SeriesDescription", "UNKNOWN")).strip()
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    photometric = getattr(ds, "PhotometricInterpretation", None)

    safe_series = (
        series_description
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    output_name = f"{dcm_path.stem}__{safe_series}__{rows}x{cols}.png"
    output_path = output_dir / output_name

    Image.fromarray(img_uint8).save(output_path)

    print()
    print("DICOM:", dcm_path)
    print("SeriesDescription:", series_description)
    print("Rows x Columns:", rows, "x", cols)
    print("Photometric:", photometric)
    print("Saved:", output_path)


def convert_folder(input_path: Path, output_dir: Path):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix.lower() == ".dcm":
        dcm_files = [input_path]
    else:
        dcm_files = sorted(input_path.rglob("*.dcm"))

    if not dcm_files:
        print(f"No DICOM files found under: {input_path}")
        return

    print(f"Found {len(dcm_files)} DICOM file(s).")

    for dcm_path in dcm_files:
        try:
            convert_dicom_file(dcm_path, output_dir)
        except Exception as e:
            print()
            print("FAILED:", dcm_path)
            print("Error:", repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="DICOM file or folder containing DICOM files",
    )
    parser.add_argument(
        "--output",
        default="outputs/debug_dicom_png",
        help="Output folder for PNG files",
    )

    args = parser.parse_args()

    convert_folder(
        input_path=Path(args.input),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()