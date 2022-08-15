import logging
from typing import List, Union

import matplotlib.pyplot as plt
import keras
import numpy as np
import cv2
from src.keypoint.models import KeypointDetector
from src.keypoint.utils import rescale_predicted_keypoints

from src.utils.dataloader import normalize, rescale_kps_from_pct
from src.utils.utilities import get_format_files


logger = logging.getLogger("bayesian_nn")


def predict_imgs_from_dir(img_dir, model: keras.Model, show_predict=True) -> np.ndarray:
    files = get_format_files(img_dir, exclude_subdirs=True)
    logger.info(f"Detected {len(files)} files")

    src_imgs = [cv2.imread(str(_img), cv2.IMREAD_GRAYSCALE) for _img in files]
    keypoints = predict_imgs(src_imgs, model)
    if show_predict:
        show_predictions(src_imgs, keypoints)

    return keypoints


def predict_imgs(
    src_imgs: Union[List[np.ndarray], np.ndarray],
    model: keras.Model,
    img_size: tuple = (224, 224),
) -> np.ndarray:
    if isinstance(src_imgs, np.ndarray):
        src_imgs = [src_imgs]

    img_arr = np.zeros((len(src_imgs), *img_size))
    img_sizes = [img.shape for img in src_imgs]
    for _i, img in enumerate(src_imgs):
        _resized_img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_AREA)
        _temp_img = normalize(_resized_img)
        img_arr[_i] = _temp_img

    img_arr = np.expand_dims(img_arr, axis=-1)

    logger.info("Predicting keypoints")
    predicted_kps = model.predict(img_arr)
    rescaled_kps = np.zeros((len(src_imgs), 14))
    for idx, _kps in enumerate(predicted_kps):
        rescaled_kps[idx] = rescale_predicted_keypoints(
            _kps, src_dim=(224, 224), target_dim=img_sizes[idx]
        )

    return rescaled_kps


def show_predictions(imgs, keypoints, labels=None):
    from src.utils.viewer import show_image_coords

    fig, axs = plt.subplots(ncols=3, nrows=len(imgs) // 3 + 1, figsize=(15, 12))

    for idx, (_img, _keypoints) in enumerate(zip(imgs, keypoints)):
        axs.flat[idx].imshow(
            np.asarray(show_image_coords(_img, _keypoints, radius=6, font_size=80))
        )
        if labels is not None:
            axs.flat[idx].text()
    plt.show()


if __name__ == "__main__":
    dir = "dataset/Inc_press_2"
    model_path = "artifacts/trained_model:v33"

    print(predict_imgs(dir, model_path))
