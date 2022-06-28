import logging
from pathlib import Path
from typing import Union, List

import numpy as np

from src.utils.constants import ACCEPTABLE_IMAGE_FORMATS
from src.utils.transforms import normalize


def get_PIL_version() -> list:
    import PIL

    return str(PIL.__version__).split(".")


def set_logger():
    logger = logging.getLogger("bayesian_nn")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_format_files(
    dir_path: Union[str, Path],
    file_formats: list = ACCEPTABLE_IMAGE_FORMATS,
    sort: bool = False,
    exclude_subdirs: bool = False,
) -> List[Path]:
    """

    Parameters
    ----------
    sort
    path : Directory to check
    file_formats: File formats to return

    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    files = [
        p.resolve() for p in dir_path.glob("**/*") if p.suffix in set(file_formats)
    ]
    if exclude_subdirs:
        files = [p for p in files if p.parent == dir_path.resolve()]
    if sort:
        return sorted(files, key=lambda x: x.name)  # sort by the file nam
    else:
        return files


def prepare_img_prediction(img_arr: np.ndarray):
    sample_img = np.expand_dims(img_arr, -1)
    sample_img = normalize(sample_img)
    return np.reshape(sample_img, (1,) + sample_img.shape).astype("float32")
