import logging
from pathlib import Path
from typing import Union

import numpy as np

from src.utils.constants import ACCEPTABLE_IMAGE_FORMATS
from src.utils.transforms import normalize


def get_PIL_version() -> list:
    import PIL

    return str(PIL.__version__).split('.')


def set_logger():
    logger = logging.getLogger('bayesian_nn')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_format_files(path: Union[str, Path], file_formats=ACCEPTABLE_IMAGE_FORMATS, sort: bool = False):
    """

    Parameters
    ----------
    sort
    path : Directory to check
    file_formats: File formats to return

    """
    if isinstance(path, str):
        path = Path(path)

    files = [p.resolve() for p in path.glob("**/*") if p.suffix in set(file_formats)]
    if sort:
        return sorted(files, key=lambda x: x.name)  # sort by the file nam
    else:
        return files


def prepare_img_prediction(img_arr: np.ndarray):
    sample_img = np.expand_dims(img_arr, -1)
    sample_img = normalize(sample_img)
    return np.reshape(sample_img, (1,) + sample_img.shape).astype('float32')
