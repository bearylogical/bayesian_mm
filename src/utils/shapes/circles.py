from src.utils.shapes.shapes import ImageGenerator
from typing import Tuple
from PIL import ImageDraw, Image
import numpy as np


class CirclesGenerator(ImageGenerator):
    """
    Circle generator class. Invoke to generate images with circles
    stochastically.
    """

    def __init__(self, save_dir=None,
                 n_images: int = 1,
                 max_circles: int = 1,
                 seed=None,
                 is_circle: bool = False,
                 is_boundary=True):
        super(CirclesGenerator).__init__(save_dir)
        self.n_images = n_images
        self.max_circles = max_circles
        self._set_rng()
        self.seed = seed
        self.is_circle = is_circle
        self.is_boundary = is_boundary

    def generate(self, save: bool = False) -> Tuple[list, list]:

        raw_images_list = []
        mask_images_list = []
        for idx in range(self.n_images):
            raw_img, mask_img = self._generate_image()

            if save:
                self.save_images(raw_img, mask_img, idx)
            else:
                raw_images_list.append(raw_img)
                mask_images_list.append(mask_img)

        return raw_images_list, mask_images_list

    def _generate_image(self) -> Tuple[Image.Image, Image.Image]:
        bg_img = self._create_image()
        temp_mask = np.zeros(self.dim)
        if self.max_circles == 1:
            num_circles = self.max_circles
        else:
            num_circles = self.rng.integers(1, self.max_circles)
        for _ in range(num_circles):
            im1 = bg_img.copy()
            prop_coords = self._generate_bounding_coords()
            while not self._check_aspect_ratio(prop_coords):
                prop_coords = self._generate_bounding_coords()  # evaluate coords to fulfil aspect ratio

            temp_img, temp_mask = draw_ellipse_mask(im1, prop_coords)
            bg_img.paste(temp_img)
            # temp_mask += temp_mask

        mask_img = Image.fromarray(temp_mask)

        return bg_img, mask_img

    def _generate_bounding_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Generates the coords for the bounds of the ellipse
        that is to be generated.
        :return: list of tuple(int, int)
        """

        x0 = self.rng.integers(low=0, high=self.dim[0])
        x1 = self.rng.integers(low=x0, high=self.dim[0])

        y0 = self.rng.integers(low=0, high=self.dim[1])
        y1 = self.rng.integers(low=y0, high=self.dim[1])

        return (x0, y0), (x1, y1)

    @staticmethod
    def _check_aspect_ratio(coords: Tuple, is_circle: bool = False) -> bool:
        (x0, y0), (x1, y1) = coords

        width = np.abs(x1 - x0)
        height = np.abs(y1 - y0)

        minor_axis, major_axis = np.min([width, height]), np.max([width, height])
        aspect_ratio = np.divide(major_axis, minor_axis, out=np.zeros_like(major_axis, dtype='float64'),
                                 where=(minor_axis != 0))

        if is_circle:
            if aspect_ratio == 1:
                return True
        elif (aspect_ratio > 1) and (aspect_ratio < 5):  # arbitrary max aspect ratio
            return True
        else:
            return False

    @staticmethod
    def _check_collision(coords: Tuple):
        raise NotImplementedError

    def save_images(self, img: Image.Image, mask: Image.Image, idx: int):
        img_fp = str(self.save_img_dir / str(idx).zfill(5)) + '.png'
        img.save(img_fp)

        mask_fp = str(self.save_segment_dir / str(idx).zfill(5)) + '.png'
        mask.save(mask_fp)

    def _create_image(self):
        return Image.new('L', self.dim, 255)  # returns a grey scale image (fully white bg)

    def _set_rng(self):
        self.rng = np.random.default_rng(self.seed)


def draw_ellipse_mask(input_image: Image.Image,
                      xy: Tuple[Tuple[int, int], Tuple[int, int]],
                      is_boundary=False) -> Tuple[Image, np.ndarray]:
    """
    Function to obtain coordinates of a generated ellipse along with a binary mask of its curvature location.
    :param is_boundary: Bool, defines whether we only want the boundaries(outlines) of the ellipse.
    :param input_image: Input Image object
    :param xy: Two points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)]
    or [x0, y0, x1, y1], where x1 >= x0 and y1 >= y0
    :return: Tuple
    """

    fill_color = None if is_boundary else np.random.randint(0, 255)  # don't include 255 because it is WHITE
    line_color = fill_color
    ImageDraw.Draw(input_image).ellipse(xy, outline=line_color, fill=fill_color)

    # MASK generation
    mask_image = np.array(input_image)
    mask_image[mask_image == fill_color] = 1
    mask_image[mask_image == 255] = 0

    # IMAGE augmentations?

    return input_image, mask_image
