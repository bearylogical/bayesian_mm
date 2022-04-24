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
from src.utils.transforms import normalize
from src.utils.utilities import get_format_files

VALID_TASKS = ["T0", "T1"]


class DataLoaderError(Exception):
    pass


class BaseDataLoader(Sequence):
    def __init__(self,
                 batch_size: int,
                 img_size: Tuple[int, int],
                 input_img_paths: List[Union[Path, str]],
                 target_paths: Union[List[Union[Path, str, dict]], Union[Path, str]],
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

        self._transform_target_data()

    def _transform_target_data(self):
        self.target_data = [json.load(open(f)) for f in self.target_data_path]
        self.targets = np.zeros((len(self.target_data), self.num_targets))
        for idx, d in enumerate(self.target_data):
            self.targets[idx] = get_keypoints_from_json(d, self.num_targets)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]

        x = self._get_input_image_data(batch_input_img_paths)
        y = self.targets[i: i + self.batch_size]

        if self.transform is not None:
            for idx, (_xi, _yi) in enumerate(zip(x, y)):
                _transformed = self.transform(image=_xi, keypoints=_yi.reshape(7, 2))
                x[idx] = _transformed['image']
                y[idx] = np.array(_transformed['keypoints']).flatten()
        #
        # if self.normalize:
        #     y = normalize_keypoints(y, batch_input_img_labels)

        return x, y


def modify_json_dict_from_keypoints(json_dict:dict, keypoints_list:Union[list, np.ndarray], num_targets)->dict:
    points = [f"p{p}" for p in range(num_targets // 2)]
    for idx, kp in enumerate(points):
        try:
            json_dict["keypoints"][kp]["x"] = keypoints_list[idx * 2]
            json_dict["keypoints"][kp]["y"] = keypoints_list[idx * 2 + 1]
        except KeyError:
            print(json_dict["file_name"])
        return json_dict

def get_keypoints_from_json(json_dict: dict, num_targets) -> list:
    """
    Gets keypoints from dictionary of JSON object to return a list.

    Parameters
    ----------
    json_dict: Dictionary obtained from JSON label file
    num_targets: Returns paired keypoints based on num of targets.
    Specifies the length of returned list by len(keypoints) = num_targets * 2.

    Returns
    -------
    A list of keypoints.
    """
    keypoints = []
    points = [f"p{p}" for p in range(num_targets // 2)]
    for kp in points:
        kp_x, kp_y = None, None
        try:
            kp_x = json_dict["keypoints"][kp]["x"]
            kp_y = json_dict["keypoints"][kp]["y"]
        except KeyError:
            print(json_dict["file_name"])
        finally:
            keypoints.append(kp_x)
            keypoints.append(kp_y)
    return keypoints


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


def get_idx_from_img_path(img_path: str) -> str:
    """
    Returns image ID from img_path
    :param img_path:
    :return:
    """
    return os.path.split(img_path)[1].split('.')[0]


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


def get_keypoint_dict_from_ls(img_shape: tuple, kp_list: list) -> dict:
    _rescaled_kps = rescale_kps_from_pct(img_shape, kp_list)
    return {f"x{idx // 2}" if idx % 2 == 0 else f"y{idx // 2}": v for idx, v in enumerate(_rescaled_kps)}


def rescale_kps_from_pct(img_shape: tuple, kp_list: list) -> list:
    h_factor, w_factor = (v / 100 for v in img_shape)
    _rescaled_kps = []
    for idx, v in enumerate(kp_list):
        if idx % 2 == 0:
            _rescaled_kps.append(v * w_factor)
        else:
            _rescaled_kps.append(v * h_factor)
    return _rescaled_kps


def get_img_target_data(img_path: Union[str, Path],
                        data_path: Union[str, Path],
                        img_size: Union[Tuple[int, int], None] = None,
                        task: str = "T1", include_idx=False) -> \
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

    data_path_ext = os.path.splitext(data_path)[-1]

    if data_path_ext == ".npz":
        img_idx = int(get_idx_from_img_path(img_path))  # should be given
        src_data = np.load(data_path)[task]
        start_idx = 0 if include_idx else 1
        field_names = src_data.dtype.names[start_idx:]
        img_props = get_target_data_from_idx(src_data, img_idx, include_idx=include_idx)
        return img, dict(zip(field_names, img_props))

    elif data_path_ext == ".json":
        kps_list = get_keypoints_from_json(json.load(open(data_path)), 14)
        return img, get_keypoint_dict_from_ls(img.shape, kps_list)

    else:
        raise Exception(f"Invalid extension of data path {data_path_ext}")


def match_image_to_target(images_path: str, targets_path: str = None,
                          target_fmt: List[str] = ACCEPTABLE_IMAGE_FORMATS,
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
    if targets_path is None:
        targets_path = get_format_files(os.path.join(images_path, "labels"), file_formats=target_fmt, sort=True)
    else:
        targets_path = get_format_files(targets_path, file_formats=target_fmt, sort=True)

    if len(image_files) != len(targets_path):
        raise DataLoaderError(
            f"Invalid number of image files ({len(image_files)}) vs label files ({len(targets_path)})")

    img_list, targets_list = [], []

    for _img, _target in zip(image_files, targets_path):
        if _img.stem == _target.stem:
            img_list.append(_img)
            targets_list.append(_target)

    return img_list, targets_list


if __name__ == "__main__":
    img_dir = "dataset/experiments/15Apr"
    labels_dir = "dataset/experiments/15Apr/labels"
    # match data and labels
    imgs, labels = match_image_to_target(img_dir, labels_dir, target_fmt=[".json"])
    kp = []
    for l in labels:
        kp.append(get_keypoints_from_json(json.load(open(l)), num_targets=14))
