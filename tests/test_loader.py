from src.utils.loader import KeyPointDataLoader, match_image_to_target, BaseDataLoader


class TestFileOperations:
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
        mock_data_loader = KeyPointDataLoader(batch_size=5,
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
