from PIL import Image
from src.utils.converter import batch_rescale_dir, parse_label_studio
from src.utils.transforms import downscale_img
from src.utils.utilities import get_format_files


class TestBulkImageConversion:

    def test_image_rescale(self, mock_disk_image):
        """
        Tests if image is loaded and rescaled correctly
        """

        assert downscale_img(fp=mock_disk_image).size == (300, 300)  # check default arg
        assert downscale_img(fp=mock_disk_image, target_size=(128, 128)).size == (128, 128)

    def test_bulk_img_filter(self, mock_bulk_image_dir):
        """
        Tests if bulk image creation is successful.

        """
        assert len(list(get_format_files(mock_bulk_image_dir))) == 30

    def test_image_bulk_conversion(self, mock_bulk_image_dir, tmp_path):
        """
        Tests if bulk image conversion is done
        """
        temp_img_dir = mock_bulk_image_dir

        temp_target_dir = tmp_path / "target_rescaled"
        temp_target_dir.mkdir()

        batch_rescale_dir(str(temp_img_dir))
        assert len(get_format_files(mock_bulk_image_dir)) == 30
        t_scaled_img = list(get_format_files(mock_bulk_image_dir))[0]
        assert Image.open(t_scaled_img).size == (300, 300)

        batch_rescale_dir(str(temp_img_dir), target_dir=str(temp_target_dir))
        assert len(list(get_format_files(temp_target_dir))) == 30

    def test_label_studio_parser(self):
        """
        Test label studio parsing capability
        """
        mock_dict = {'id': 55,
                     "file_upload" : "some_file.png",
                     'annotations': [{'id': 56,
                                      'completed_by': 1,
                                      'result': [{'original_width': 300,
                                                  'original_height': 213,
                                                  'image_rotation': 0,
                                                  'value': {'x': 1.3333333333333335,
                                                            'y': 40.375586854460096,
                                                            'width': 0.3418803418803419,
                                                            'keypointlabels': ['p0']},
                                                  'id': '_bGIh3Rygs',
                                                  'from_name': 'kp-1',
                                                  'to_name': 'img-1',
                                                  'type': 'keypointlabels',
                                                  'origin': 'manual'},
                                                 {'original_width': 300,
                                                  'original_height': 213,
                                                  'image_rotation': 0,
                                                  'value': {'x': 1.3333333333333335,
                                                            'y': 40.375586854460096,
                                                            'width': 0.3418803418803419,
                                                            'keypointlabels': ['p1']},
                                                  'id': '_bGIh3Rygs',
                                                  'from_name': 'kp-1',
                                                  'to_name': 'img-1',
                                                  'type': 'keypointlabels',
                                                  'origin': 'manual'}]}]}
        expected_result = {
            "file_name" : "some_file.png",
            "original_width": 300,
            "original_height": 213,
            "keypoints": {
                "p0": dict(x=1.3333333333333335, y=40.375586854460096, width=0.3418803418803419),
                "p1": dict(x=1.3333333333333335, y=40.375586854460096, width=0.3418803418803419)
            }}
        result = parse_label_studio(mock_dict)
        assert result == expected_result
