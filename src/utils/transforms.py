import numpy as np


def normalize(input_image: np.ndarray, max_val: float = 255.0):
    input_image = input_image.astype("float32") / max_val
    return input_image


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