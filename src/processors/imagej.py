import numpy as np

from src.utils.constants import PRESSURE_CONSTANT_MULTIPLIER


def load_keypoints_pressures(coord_fp: str = None,
                             pressure_fp: str = None,
                             normalize: bool = True,
                             pressure_col_idx: int = 1,
                             num_points_per_image: int = 7):
    x_coords, y_coords = load_imagej_data(coord_fp, normalize, num_points_per_image=num_points_per_image)
    pressures = load_pressure_data(pressure_fp, pressure_col_idx=pressure_col_idx)

    if len(x_coords) != len(pressures):
        print(
            "Number of image slices not equal to supplied pressures. Data will be truncated accordingly."
        )
        min_slices = min(len(x_coords), len(pressures))
        x_coords = x_coords[: (min_slices * num_points_per_image)]
        y_coords = y_coords[: (min_slices * num_points_per_image)]
        pressures = pressures[:min_slices]

    return x_coords, y_coords, pressures


def load_imagej_data(coord_fp: str = None,
                     normalize: bool = True,
                     num_points_per_image=7):
    """
    Parameters
    ----------
    num_points_per_image: Number of keypoints that define an image
    coord_fp : File path of coordinates
    pressure_fp : File path of CSV file.
    coord_format : For coordinate files in different formats. Currently only "ImageJ" is supported.
    normalize: Normalize all the points within a specific experiment. Each experiment
    is characterized by multiple images, with each having a set of points. Normalize ensures that
    all the points are consistent throughout the experiments
    pressure_col_idx: Row to read off the column pressure.

    Returns
    -------

    """

    with open(coord_fp) as f:
        header = f.readline().split()
        coord_dtype = {
            "names": tuple(header),
            "formats": (("i4",) * len(header)),
        }

    coord_data = np.loadtxt(coord_fp, skiprows=1, dtype=coord_dtype)
    num_images = len(coord_data) // num_points_per_image

    x_coords = coord_data["x"].reshape(-1, num_points_per_image)
    y_coords = coord_data["y"].reshape(-1, num_points_per_image)

    if normalize:
        x_coords = normalize_keypoint(x_coords)
        y_coords = normalize_keypoint(y_coords)

    # self.capillary_points = [CapillaryPoints()]
    return x_coords, y_coords


def load_pressure_data(pressure_fp: str = None, pressure_col_idx: int = 1, ):
    pressures = (
            np.loadtxt(pressure_fp, skiprows=1, dtype="f4", usecols=pressure_col_idx)
            * PRESSURE_CONSTANT_MULTIPLIER
    )

    return pressures


def normalize_keypoint(coords, ref_idx=0):
    return coords - coords[:, ref_idx][:, np.newaxis]
