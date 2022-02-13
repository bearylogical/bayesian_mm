import os

from tensorflow.keras.utils import Sequence
from src.utils.constants import ACCEPTABLE_IMAGE_FORMATS, \
    ACCEPTABLE_SEGMENTATION_FORMATS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import List, Tuple, Union
from PIL import Image
from pathlib import Path
VALID_TASKS = ["T0", "T1"]


# Code modified from Image Segmentation Keras library
# Divam Gupta, Rounaq Wala , Marius Juston, JaledMC
# https://github.com/divamgupta/image-segmentation-keras


class DataLoaderError(Exception):
    pass


class BaseDataLoader(Sequence):
    def __init__(self,
                 batch_size,
                 img_size,
                 input_img_paths,
                 target_paths,
                 is_rgb=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_paths
        self.channels = 3 if is_rgb else 1
        self.color_mode = "rgb" if is_rgb else "grayscale"
        self._is_rgb = is_rgb

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def _get_input_image_data(self, batch_input_img_paths):
        x = np.zeros((self.batch_size,) + self.img_size + (self.channels,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, color_mode=self.color_mode, target_size=self.img_size)
            img = img_to_array(img)
            img = normalize(img)  # normalise inputs such that [0,1]
            x[j] = img if self._is_rgb else np.expand_dims(img, axis=0)

        return x


class BaseRegressionDataLoader(BaseDataLoader):
    def __init__(self, batch_size, img_size, input_img_paths, target_paths, num_targets=4,task: str = "T1", fields=None):
        super().__init__(batch_size, img_size, input_img_paths, target_paths)
        self.target_data_path = target_paths
        self.num_targets = num_targets
        self.fields = fields
        self.task = task

        self._check_task()

    def _check_task(self):
        if self.task not in VALID_TASKS:
            raise AttributeError(f"Task: {self.task}Not a valid task")

    def _get_target_data(self, batch_input_img_idx, fields):
        y_data = np.load(self.target_data_path)[self.task]
        y = np.zeros((self.batch_size, self.num_targets))
        for i, img_idx in enumerate(batch_input_img_idx):
            y[i] = get_target_data_from_idx(y_data, img_idx, fields=fields)
        return y


class RegressionDataLoaderT0(BaseRegressionDataLoader):
    def __init__(self, batch_size, img_size, input_img_paths, target_paths, num_targets=4, task=None, fields=None):
        super().__init__(batch_size, img_size, input_img_paths, target_paths, fields, task)
        self.num_targets = num_targets

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_input_img_idx = [get_idx_from_img_path(f) for f in batch_input_img_paths]

        x = self._get_input_image_data(batch_input_img_paths)
        y = self._get_target_data(batch_input_img_idx, fields=self.fields)
        return x, y


class RegressionDataLoaderT1(BaseRegressionDataLoader):
    def __init__(self, batch_size, img_size, input_img_paths, target_paths, num_targets=4,task=None, fields=None,
                 normalize: bool = True ):
        super().__init__(batch_size, img_size, input_img_paths, target_paths, fields, task)
        self.num_targets = num_targets
        self.normalize = normalize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_input_img_idx = [get_idx_from_img_path(f) for f in batch_input_img_paths]

        x = self._get_input_image_data(batch_input_img_paths)
        y = self._get_target_data(batch_input_img_idx, fields=self.fields)
        if self.normalize:
            y = normalize(y, float(self.img_size))
        return x, y


class SegmentDataLoader(BaseDataLoader):
    def __init__(self, batch_size, img_size, input_img_paths, target_paths):
        super().__init__(batch_size, img_size, input_img_paths, target_paths)

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = self._get_input_image_data(batch_input_img_paths)

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


def normalize(input_image: np.ndarray, max_val: float = 255.0):
    input_image = input_image.astype("float32") / max_val
    return input_image


def _get_images_from_dir(image_dir: str, sort=True) -> List[Tuple[str, str, str]]:
    image_list = []

    for dir_entry in os.listdir(image_dir):
        file_name, file_extension = os.path.splitext(dir_entry)
        if os.path.isfile(os.path.join(image_dir, dir_entry)) and \
                file_extension in ACCEPTABLE_IMAGE_FORMATS:
            image_list.append((file_name, file_extension,
                               os.path.join(image_dir, dir_entry)))

    if sort:
        return sorted(image_list, key=lambda x: x[0])  # sort by the file nam
    else:
        return image_list


def get_image_paths_from_dir(image_dir: str) -> List:
    """
    Returns a list of image path in the proposed image directory with an */images/ parent dir
    :param image_dir:
    :return:
    """
    img_list = _get_images_from_dir(image_dir)

    return [img[2] for img in img_list]


def get_idx_from_img_path(img_path: str) -> int:
    """
    Returns image ID from img_path
    :param img_path:
    :return:
    """
    return int(os.path.split(img_path)[1].split('.')[0])


def get_target_data_from_idx(data: np.ndarray, img_idx: int, include_idx=False,
                             fields: Union[List[str], None] = None) -> Tuple:
    """
    Returns a tuple containing the target data
    :param data: Structured np.ndarray as input
    :param img_idx: Index of image that data is being requested for
    :param include_idx: Bool. If true, includes the idx value which is always at the start of the array.
    :param fields: Data fields to include. Defaults to None. When include_idx is included, all the rows are returned as
    a Tuple.
    :return:
    """
    # TODO: this is a very course data getter. needs some validation logic
    mask = data['idx'] == img_idx
    f_data = data[mask]
    if fields is None:
        start_idx = 0 if include_idx else 1
        return f_data.item()[start_idx:]
    else:
        return f_data[fields].item()


def get_img_target_data(img_path: Union[str, Path], data_path: Union[str, Path], img_size: Union[Tuple[int, int] , None]=None, task: str = "T1", include_idx=False) -> \
        Tuple[Image.Image, dict]:
    """
    Returns a tuple containing the Image as a PIL instance and a dictionary
    with the field property as key and its associated value.
    :param task: Refer to Prediction Task document
    :param img_size: Size of the image (Int, Int)
    :param img_path: Path of the image file
    :param data_path: Path to either the npz or npy file containing the data
    :param include_idx: Include the id col
    :return:
    """
    img = load_img(img_path, target_size=img_size)  # should be given
    img_idx = get_idx_from_img_path(img_path)  # should be given

    data_path_ext = os.path.splitext(data_path)[-1]

    if data_path_ext == ".npz":
        src_data = np.load(data_path)[task]
    else:
        src_data = np.load(data_path)

    start_idx = 0 if include_idx else 1
    field_names = src_data.dtype.names[start_idx:]
    img_props = get_target_data_from_idx(src_data, img_idx, include_idx=include_idx)

    return img, dict(zip(field_names, img_props))


def _get_img_seg_path(src_dir: str, img_dir_name: str = "images", segment_dir_name: str = "segment"):
    """
    Gets the directory of the image and segments
    :param src_dir:
    :param img_dir_name:
    :param segment_dir_name:
    :return:
    """
    im_path = ""
    segment_path = ""

    for dir_entry in os.listdir(src_dir):
        dpath = os.path.join(src_dir, dir_entry)
        if os.path.isdir(dpath):
            if dir_entry == img_dir_name:
                im_path = dpath
            elif dir_entry == segment_dir_name:
                segment_path = dpath
            else:
                raise FileNotFoundError

    return im_path, segment_path


def get_pairs_from_paths(images_path: str, segs_path: str, ignore_non_match: bool = True) -> Tuple[List, List]:
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    image_files = []
    seg_files = []

    image_files = _get_images_from_dir(images_path)
    seg_files = _get_images_from_dir(segs_path)

    if len(image_files) != len(seg_files):
        raise DataLoaderError(f"Invalid number of image files ({len(image_files)}) vs segment files ({seg_files})")

    img_list, seg_list = [], []
    for _img, _seg in zip(image_files, seg_files):
        if _img[0] == _seg[0]:
            img_list.append(_img[2])
            seg_list.append(_seg[2])

    return img_list, seg_list


if __name__ == "__main__":
    generation_date = "20220210"
    demo_img_path = get_image_paths_from_dir(f"dataset/{generation_date}/images")[3]
    t_data_path = f"dataset/{generation_date}/images/targets.npy"
    target_info = get_img_target_data(demo_img_path, t_data_path)
