import mock
from src.keypoint.models.regression.cnn_regression import BaseKeypointModel
from src.keypoint.models import KeypointDetector
import cv2

from src.utils.utilities import get_format_files


class TestKeypointDetector:
    def test_keypoint_object(self, mock_keypoint_detector: KeypointDetector):
        assert mock_keypoint_detector.num_target == 14
        assert mock_keypoint_detector.img_size == (224, 224)

    def test_keypoint_model(self, mock_keypoint_detector: KeypointDetector):
        assert mock_keypoint_detector.model is not None
        assert isinstance(mock_keypoint_detector.model, BaseKeypointModel)

    def test_predict_single_image(
        self, mock_keypoint_detector: KeypointDetector, mock_disk_image
    ):
        img = cv2.imread(str(mock_disk_image), cv2.IMREAD_GRAYSCALE)
        keypoint = mock_keypoint_detector.predict(img)

        assert len(keypoint) == 1
        assert keypoint.shape == (1, 14)

    def test_predict_multiple_images(
        self, mock_keypoint_detector: KeypointDetector, mock_bulk_image_dir
    ):
        bulk_images = get_format_files(mock_bulk_image_dir)
        imgs = [cv2.imread(str(_img), cv2.IMREAD_GRAYSCALE) for _img in bulk_images]
        keypoints = mock_keypoint_detector.predict(imgs)

        assert len(keypoints) == len(imgs)
        assert keypoints.shape == (len(imgs), 14)
        
    def test_mc_dropout(
        self, mock_keypoint_detector: KeypointDetector, mock_bulk_image_dir
    )
        pass
