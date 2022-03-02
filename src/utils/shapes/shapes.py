import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Type
from time import strftime
from scipy.special import comb
import shutil
import logging
from collections.abc import Sequence

logger = logging.getLogger()


class ImageGenerator:
    """
    Base class for image generator, defaults to create a sample image 128x128
    """

    def __init__(self, save_dir: Union[None, Path] = None,
                 dim: Tuple[int, int] = (128, 128),
                 seed: Union[None, int] = None,
                 is_segment: bool = False,
                 is_train_split: bool = True,
                 train_test_ratio: Union[float, None] = None,
                 force_images: bool = False):
        # TODO: Move this implementation to another method instead of running on initialisation
        if save_dir is None:  # initialise save dir as current date ISO8601 YYYYMMDD format
            temp_dir = Path.cwd() / 'dataset' / strftime("%Y%m%d")
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = temp_dir
        elif isinstance(save_dir, Path):
            if save_dir.is_dir():
                self.save_dir = save_dir
        else:
            raise AttributeError

        if force_images:
            print(f"Force images set to true, deleting old files in {self.save_dir}")
            shutil.rmtree(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if is_segment:
            self.save_segment_dir = self.save_dir / 'segment'
            self.save_segment_dir.mkdir(exist_ok=True)

        self.save_img_dir = self.save_dir / 'images'
        self.save_img_dir.mkdir(exist_ok=True)

        if is_train_split or train_test_ratio is not None:
            self.train_dir = self.save_img_dir / 'train'
            self.test_dir = self.save_img_dir / 'test'

            self.train_dir.mkdir(exist_ok=True)
            self.test_dir.mkdir(exist_ok=True)
            self.train_test_ratio = 0.8 if train_test_ratio is None else train_test_ratio
            self.is_train_split = is_train_split

        self.dim = dim

        self.seed = seed  # if none will get from OS.

    def _check_dir(self, **kwargs):
        pass

    def _generate_image(self, **kwargs):
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError


def get_bezier_parameters(x, Y, degree=2):
    """ Least square qbezier fit using penrose pseudo-inverse.

    Parameters:

    x: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(x) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(x) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(x)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(x))
    M = bmatrix(T)
    points = np.array(list(zip(x, Y)))
    return least_square_fit(points, M).tolist()


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, n_times=50):
    """
       Given a set of control points, return the
       Bézier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ...
                 [Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See https://processingjs.nihongoresources.com/bezierinfo/
    """

    n_points = len(points)
    x_points = np.array([p[0] for p in points])
    y_points = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, n_times)

    polynomial_array = np.array([bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)])

    x_vals = np.dot(x_points, polynomial_array)
    y_vals = np.dot(y_points, polynomial_array)

    return x_vals, y_vals


## keypoint scale from Albumentations library

def apply_keypoints_scale(keypoints: Sequence[Sequence[float]],
                          img_size: Tuple[int, int],
                          target_size: Tuple[int, int]):
    return [apply_keypoint_scale(keypoint, img_size=img_size, target_size=target_size) for keypoint in
            keypoints]


def apply_keypoint_scale(keypoint: Sequence[float],
                         img_size: Tuple[int, int],
                         target_size: Tuple[int, int]):
    scale_x, scale_y = get_scale_factor(img_size, target_size)
    return keypoint_scale(keypoint, scale_x, scale_y)


def get_scale_factor(img_size: Tuple[int, int], target_size: Tuple[int, int]):
    width, height = img_size
    target_width, target_height = target_size
    scale_x = target_width / width
    scale_y = target_height / height
    return scale_x, scale_y


def keypoint_scale(keypoint: Sequence[float],
                   scale_x: float,
                   scale_y: float,
                   dtype: Union[Type[int], Type[float]] = int):
    """Scales a keypoint by scale_x and scale_y.
    Args:
        keypoint (tuple): A keypoint `(x, y)`.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.
        dtype: Return type of output
    Returns:
        A keypoint `(x, y, angle, scale)`.
    """
    x, y = keypoint[:2]
    if isinstance(x, int) & isinstance(y, int):
        return int(x * scale_x), int(y * scale_y)
    else:
        return x * scale_x, y * scale_y
