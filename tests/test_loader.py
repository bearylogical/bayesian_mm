import json

import numpy as np

from src.utils.loader import KeyPointDataLoader, \
    match_image_to_target, \
    BaseDataLoader, \
    get_idx_from_img_path,\
    get_keypoints_from_json,\
    get_img_target_data


class TestFileOperations:

    def test_get_img_id_from_path(self):
        assert get_idx_from_img_path("sample.py") == "sample"
        assert get_idx_from_img_path("1.jpg") == "1"

    def test_get_keypoints_from_json(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       str(mock_label_dir),
                                                       target_fmt=[".json"])
        mock_labels = [json.load(open(f)) for f in mock_labels]
        res = get_keypoints_from_json(mock_labels[0], 14)
        assert len(res) == 14
        assert res == [1., 41., 2. , 42., 3., 43., 4., 44., 5., 45., 6., 46. , 7., 47.]

    def test_get_image_data_json(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       str(mock_label_dir),
                                                       target_fmt=[".json"])
        res = get_img_target_data(mock_imgs[0], mock_labels[0])
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (600, 600)
        assert len(res[1]) == 14
        assert res[1]["x0"] == 6.
        assert res[1]["y3"] == 264.
        assert res[1]["x4"] == 30.

    def test_get_image_data_json_defaults(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       target_fmt=[".json"])
        res_default = get_img_target_data(mock_imgs[0], mock_labels[0])
        assert isinstance(res_default[0], np.ndarray)
        assert res_default[0].shape == (600, 600)
        assert len(res_default[1]) == 14
        assert res_default[1]["x0"] == 6.
        assert res_default[1]["y3"] == 264.
        assert res_default[1]["x4"] == 30.

    def test_input_target_match(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       str(mock_label_dir),
                                                       target_fmt=[".json"])
        assert mock_labels[0].stem == mock_imgs[0].stem

class TestBaseDataLoader:
    def test_base_data_loader_image_load(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       str(mock_label_dir),
                                                       target_fmt=[".json"])
        mock_data_loader = BaseDataLoader(batch_size=5,
                                              img_size=(128, 128),
                                              transform=None,
                                              input_img_paths=mock_imgs,
                                              target_paths=mock_labels)
        mock_target = mock_data_loader._get_input_image_data(mock_imgs[:5])
        target_shape = (5, 128, 128, 1)
        assert mock_target.shape == target_shape



class TestKeypointLoader:
    def test_keypoint_loading(self, mock_img_keypoint_label):
        mock_img_dir, mock_label_dir = mock_img_keypoint_label
        mock_imgs, mock_labels = match_image_to_target(str(mock_img_dir),
                                                       str(mock_label_dir),
                                                       target_fmt=[".json"])
        mock_keypoint_loader = KeyPointDataLoader(batch_size=5,
                                                  img_size=(128, 128),
                                                  transform=None,
                                                  input_img_paths=mock_imgs,
                                                  target_paths=mock_labels)

        mock_X, mock_y = mock_keypoint_loader.__getitem__(0)
        target_X_shape = (5, 128, 128, 1)
        target_y_shape = (5, 14)
        assert mock_X.shape == target_X_shape
        assert mock_y.shape == target_y_shape

    def test_keypoint_transforms(self):
        pass
