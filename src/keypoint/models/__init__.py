from typing import Union, List
import logging

import cv2
import numpy as np
from keras.models import load_model, Model
from keras import layers
from src.keypoint.models.regression.uncertainty import MCDropoutRegression
from src.keypoint.utils import rescale_predicted_keypoints

from src.utils.dataloader import normalize

logger = logging.getLogger("bayesian_nn")


class KeypointDetector:
    """
        Base class for keypoint detection   
    """

    def __init__(self, num_target: int = 14, img_size: tuple = (224, 224)) -> None:
        """_summary_

        Parameters
        ----------
        num_target : int, optional
            _description_, by default 14
        img_size : tuple, optional
            _description_, by default (224, 224)
        """
        self.num_target = num_target
        self.img_size = img_size
        self.model: Model = None

    def load(self, model_path):
        self.model = load_model(model_path)

    def _predict(self, img_arr: np.ndarray, img_sizes: List[tuple]):
        predicted_kps = self.model.predict(img_arr)
        rescaled_kps = np.zeros_like(predicted_kps)
        for idx, _kps in enumerate(predicted_kps):
            rescaled_kps[idx] = rescale_predicted_keypoints(
                _kps, src_dim=(224, 224), target_dim=img_sizes[idx]
            )

        return rescaled_kps

    def _check_mc_dropout_model(self):
        # this is not optimal, we should be checking for the model configuration.
        return True if "mc_dropout_regression" in self.model.name else False

    def predict(
        self, src_imgs: Union[List[np.ndarray], np.ndarray], num_passes: int = None
    ):

        assert self.model is not None, "No Model defined!"

        if isinstance(src_imgs, np.ndarray):
            src_imgs = [src_imgs]

        num_imgs = len(src_imgs)
        img_arr = np.zeros((num_imgs, *self.img_size))
        img_sizes: List[tuple] = [img.shape for img in src_imgs]
        for _i, img in enumerate(src_imgs):
            _resized_img = cv2.resize(
                img, dsize=self.img_size, interpolation=cv2.INTER_AREA
            )
            _temp_img = normalize(_resized_img)
            img_arr[_i] = _temp_img

        img_arr = np.expand_dims(img_arr, axis=-1)

        logger.info("Predicting keypoints")

        # check for dropout model
        if all([num_passes is not None, self._check_mc_dropout_model()]):
            assert num_passes > 0
            logger.info(
                f"MC Dropout model detected, Running MC dropout for {num_passes} passes"
            )
            predicted_kps = np.zeros(shape=(num_passes * num_imgs, self.num_target))
            for idx in range(num_passes):
                predicted_kps[num_imgs * idx : num_imgs * (idx + 1), :] = self._predict(
                    img_arr, img_sizes
                )
        else:
            predicted_kps = self._predict(img_arr, img_sizes)

        return predicted_kps

    def _show_predictions(self, img, keypoints):
        return NotImplementedError

    def _save_predictions(self, img, keypoints):
        return NotImplementedError

