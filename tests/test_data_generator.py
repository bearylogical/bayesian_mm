import numpy as np
from src.utils.shapes import draw_ellipse_mask
from PIL.Image import Image


class TestDataGenerator:
    def test_ellipse_draw(self, mock_image):
        image, mask = draw_ellipse_mask(mock_image, ((1, 1), (10, 10)))

        assert isinstance(image, Image)
        assert isinstance(mask, np.ndarray)

    def test_circles_generator(self, mock_circles_generator):
        """
        Test if the circles generator is working
        :param mock_circles_generator: CirclesGenerator object
        :return:
        """
        img, mask = mock_circles_generator._generate_image()

        assert isinstance(img, Image)
        assert isinstance(mask, Image)
        # TODO : check for ONLY 0 and 1 in the mask

    def test_valid_coords(self, mock_circles_generator):
        mock_coords_invalid = (0, 1), (0, 0)
        mock_coords_valid = (0,1), (4, 0)

        assert not mock_circles_generator._check_aspect_ratio(mock_coords_invalid)
        assert mock_circles_generator._check_aspect_ratio(mock_coords_valid)
