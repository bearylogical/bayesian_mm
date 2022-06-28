import numpy as np
from src.utils.dataloader import rescale_kps_from_pct


def get_xy(keypoints: np.ndarray):
    if keypoints.shape[1] != 14:
        raise Exception("Invalid number of keypoints. Needs to be 7 xy pairs")

    x_coords = keypoints[:, ::2].reshape(1, -1).flatten()
    y_coords = keypoints[:, 1::2].reshape(1, -1).flatten()
    return x_coords, y_coords


def rescale_predicted_keypoints(
    predicted_kps: np.ndarray, src_dim: tuple, target_dim: tuple
) -> np.ndarray:
    """Rescale predicted keypoints from a source image to a target image

    Parameters
    ----------
    predicted_kps : np.ndarray
        Numpy array containing the predicted keypoint
    src_dim : tuple
        Size of source image (w, h)
    target_dim : tuple
        Size of target image (w, h)

    Returns
    -------
    np.ndarray
        Numpy array containing the rescaled keypoint
    """
    a_ratio = target_dim[0] / src_dim[0], target_dim[1] / src_dim[1]
    resized_kps = rescale_kps_from_pct(src_dim, [predicted_kps])[0]
    resized_kps[::2] = resized_kps[::2] * a_ratio[1]
    resized_kps[1::2] = resized_kps[1::2] * a_ratio[0]

    return resized_kps
