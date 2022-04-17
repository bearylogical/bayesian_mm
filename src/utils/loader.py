import os

from tensorflow.keras.utils import Sequence
from src.utils.constants import ACCEPTABLE_IMAGE_FORMATS
from src.utils.augmentations import default_aug
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from typing import List, Tuple, Union
from pathlib import Path
import json
import albumentations as A
import cv2
from src.utils.transforms import normalize, normalize_keypoints
from src.utils.utilities import get_format_files

VALID_TASKS = ["T0", "T1"]


# Code modified from Image Segmentation Keras library
# Divam Gupta, Rounaq Wala , Marius Juston, JaledMC
# https://github.com/divamgupta/image-segmentation-keras


class DataLoaderError(Exception):
    pass


class BaseDataLoader(Sequence):
    def __init__(self,
                 batch_size: int,
                 img_size: Tuple[int, int],
                 input_img_paths: List[Union[Path, str]],
                 target_paths: Union[List[Union[Path, str]], Union[Path, str]],
                 is_rgb: bool = False,
                 transform=None):
        """
        Base class for Data Loaders

        :param batch_size: Number of batches.
        :param img_size: height x width  of input img
        :param input_img_paths: List of image paths
        :param target_paths: Can be a list of target paths (for classification/segmentation) or a single target path (for regression)
        :param is_rgb:
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_paths
        self.channels = 3 if is_rgb else 1
        self.color_mode = "rgb" if is_rgb else "grayscale"
        self._is_rgb = is_rgb
        self.transform = transform
        self.scale_img = None

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def _get_input_image_data(self, batch_input_img_paths):
        x = np.zeros((self.batch_size,) + self.img_size + (self.channels,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            if isinstance(path, Path):
                path = str(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.scale_img is None:
                self.scale_img = img.shape[0] / self.img_size[0]
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_AREA)
            img = normalize(img)  # normalise inputs such that [0,1]
            x[j] = img if self._is_rgb else np.expand_dims(img, axis=-1)

        return x


class BaseRegressionDataLoader(BaseDataLoader):
    def __init__(self, batch_size, img_size, input_img_paths, target_paths, num_targets=None, task: str = "T1",
                 fields=None):
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
        if self.num_targets is None:
            print('No number of targets supplied, inferring from data source')
            fields = [name for name in y_data.dtype.names if name != 'idx']
            print(f'{len(fields)} targets identified.')
            self.num_targets = len(fields)

        y = np.zeros((self.batch_size, self.num_targets))
        for i, img_idx in enumerate(batch_input_img_idx):
            _y = get_target_data_from_idx(y_data, img_idx, fields=fields)
            if self.scale_img is not None:
                y[i] = [_ty / self.scale_img for _ty in _y]
            else:
                y[i] = _y

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
    def __init__(self, batch_size,
                 img_size,
                 input_img_paths,
                 target_paths,
                 num_targets=None,
                 task=None,
                 fields=None,
                 normalize: bool = False,
                 transform: Union[A.Compose, None] = default_aug()):
        super().__init__(batch_size, img_size, input_img_paths, target_paths, fields, task)
        self.num_targets = num_targets
        self.normalize = normalize
        self.transform = transform

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_input_img_idx = [get_idx_from_img_path(f) for f in batch_input_img_paths]

        x = self._get_input_image_data(batch_input_img_paths)
        y = self._get_target_data(batch_input_img_idx, fields=self.fields)

        if self.transform is not None:
            for idx, (_xi, _yi) in enumerate(zip(x, y)):
                _transformed = self.transform(image=_xi, keypoints=_yi.reshape(7, 2))
                x[idx] = _transformed['image']
                y[idx] = np.array(_transformed['keypoints']).flatten()

        if self.normalize:
            if self.img_size[0] != self.img_size[1]:
                raise DataLoaderError(f"Image size not equal:"
                                      f"{self.img_size[0]} "
                                      f"not equal to {self.img_size[1]}")
            y = normalize(y, float(self.img_size[0]))

        return x, y


class KeyPointDataLoader(BaseDataLoader):
    def __init__(self, batch_size,
                 img_size,
                 input_img_paths,
                 target_paths,
                 num_targets=14,
                 normalize: bool = True,
                 transform: Union[A.Compose, None] = default_aug()):
        super().__init__(batch_size, img_size, input_img_paths, target_paths)
        self.target_data_path = target_paths
        self.num_targets = num_targets
        self.normalize = normalize
        self.transform = transform

    def _get_target_data(self, target_paths: List[Path]):

        y = np.zeros((self.batch_size, self.num_targets))
        for i, target in enumerate(target_paths):
            with open(target) as _f:
                _y = json.load(_f)
            keypoints = []
            points = [f"p{p}" for p in range(self.num_targets // 2)]
            for kp in points:
                keypoints.append(_y["keypoints"][kp]["x"])
                keypoints.append(_y["keypoints"][kp]["y"])
            if self.scale_img is not None:
                y[i] = [_ty / self.scale_img for _ty in keypoints]
            else:
                y[i] = keypoints

        return y

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_input_img_labels = self.target_data_path[i: i + self.batch_size]

        x = self._get_input_image_data(batch_input_img_paths)
        y = self._get_target_data(batch_input_img_labels)

        if self.transform is not None:
            for idx, (_xi, _yi) in enumerate(zip(x, y)):
                _transformed = self.transform(image=_xi, keypoints=_yi.reshape(7, 2))
                x[idx] = _transformed['image']
                y[idx] = np.array(_transformed['keypoints']).flatten()

        if self.normalize:
            y = normalize_keypoints(y, batch_input_img_labels)

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


def prepare_img_prediction(img_arr: np.ndarray):
    sample_img = np.expand_dims(img_arr, -1)
    sample_img = normalize(sample_img)
    return np.reshape(sample_img, (1,) + sample_img.shape).astype('float32')


def _get_images_from_dir(image_dir: Union[str, Path], sort=True) -> List[Tuple[str, str, str]]:
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)

    image_list = get_format_files(image_dir)
    if sort:
        return sorted(image_list, key=lambda x: x[0])  # sort by the file nam
    else:
        return image_list


def get_image_paths_from_dir(image_dir: Union[str, Path]) -> List[Path]:
    """
    Returns a list of image path in the proposed image directory with an */images/ parent dir
    :param image_dir:
    :return:
    """
    return get_format_files(image_dir)


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


def get_img_target_data(img_path: Union[str, Path], data_path: Union[str, Path],
                        img_size: Union[Tuple[int, int], None] = None, task: str = "T1", include_idx=False) -> \
        Tuple[np.ndarray, dict]:
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
    if isinstance(img_path, Path):
        img_path = str(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_size is not None:  # should be given
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
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


def match_image_to_target(images_path: str, targets_path: str,
                          target_fmt:List[str]=ACCEPTABLE_IMAGE_FORMATS,
                          ignore_non_match: bool = True) -> Tuple[List, List]:
    """
    Find all the images from the images_path directory and
    the segmentation images from the segs_path directory
    while checking integrity of data

    :param images_path:
    :param segs_path:
    :param ignore_non_match:
    :return:
    """

    image_files = get_format_files(images_path, sort=True)
    targets_path = get_format_files(targets_path, file_formats=target_fmt, sort=True)

    if len(image_files) != len(targets_path):
        raise DataLoaderError(f"Invalid number of image files ({len(image_files)}) vs segment files ({targets_path})")

    img_list, targets_list = [], []

    for _img, _target in zip(image_files, targets_path):
        if _img.stem == _target.stem:
            img_list.append(_img)
            targets_list.append(_target)

    return img_list, targets_list


if __name__ == "__main__":
    generation_date = "20220303"
    demo_img_path = get_image_paths_from_dir(f"dataset/{generation_date}/images")[0]
    t_data_path = f"dataset/{generation_date}/images/targets.npy"
    target_info = get_img_target_data(demo_img_path, t_data_path)
