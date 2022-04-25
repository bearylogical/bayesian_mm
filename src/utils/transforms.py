from pathlib import Path
import json
from typing import List, Tuple
import numpy as np
from PIL import Image


def normalize(input_image: np.ndarray, max_val: float = 255.0):
    input_image = input_image.astype("float32") / max_val
    return input_image


def normalize_label_studio_keypoints(keypoints: np.ndarray, input_label_paths: List[Path]):
    for idx, (k, p) in enumerate(zip(keypoints, input_label_paths)):
        with open(p) as kp_file:
            kp_props = json.load(kp_file)
        keypoints[idx][::2] = k[::2] / kp_props["original_width"]  # apply norm on x by dividing image width
        keypoints[idx][1::2] = k[1::2] / kp_props["original_height"]  # apply norm on y by dividing image height

    return keypoints


def reject_outliers(data: np.ndarray, m=2) -> np.ndarray:
    """
    Rejects outliers that are at least 2 standard deviations away.
    :param data:
    :param m: Number of standard deviations to reject
    :return:
    """
    mask = np.abs(data - np.mean(data, axis=0)) <= m * np.std(data, axis=0)

    if data.ndim == 1:
        return data[mask]
    else:
        return data[np.all(mask, axis=1)]


def divide_by_zero(num, den):
    return np.divide(num, den, out=np.zeros_like(num), where=(den) != 0)


def normalise_bands(bands, img_size=(2880, 2048)):
    if (bands.ndim == 2) and (len(img_size) == 2):
        r_band, l_band = bands[:, 0], bands[:, 1]
    elif bands.ndim == 1:
        r_band, l_band = bands[0], bands[1]

    else:
        raise Exception('Wrong length for bands')

    return np.array([r_band / img_size[1], l_band / img_size[0]])


def downscale_img(fp: Path, target_size: Tuple[int, int] = (300, 300)) -> Image.Image:
    img = Image.open(str(fp))
    img.thumbnail(target_size)

    return img
