import os

from tensorflow.keras.utils import Sequence
from constants import ACCEPTABLE_IMAGE_FORMATS, \
    ACCEPTABLE_SEGMENTATION_FORMATS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from typing import List, Tuple
# Code modified from Image Segmentation Keras library
# Divam Gupta, Rounaq Wala , Marius Juston, JaledMC
# https://github.com/divamgupta/image-segmentation-keras


class DataLoaderError(Exception):
    pass


class DataLoader(Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y

# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     input_mask -= 1
#     return input_image, input_mask


def _get_images_from_dir(image_dir: str, sort=True) -> List[Tuple[str, str, str]]:
    image_list = []

    for dir_entry in os.listdir(image_dir):
        file_name, file_extension = os.path.splitext(dir_entry)
        if os.path.isfile(os.path.join(image_dir, dir_entry)) and \
                file_extension in ACCEPTABLE_IMAGE_FORMATS:
            image_list.append((file_name, file_extension,
                               os.path.join(image_dir, dir_entry)))

    if sort:
        return sorted(image_list, key=lambda x : x[0]) # sort by the file nam
    else:
        return image_list


def _get_img_seg_path(src_dir: str, img_dir_name:str = "images", segment_dir_name:str = "segment"):
    """
    Gets the directory of the
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
            if dir_entry  == img_dir_name:
                im_path = dpath
            elif dir_entry == segment_dir_name:
                segment_path = dpath
            else:
                raise FileNotFoundError

    return im_path, segment_path


def _get_pairs_from_paths(images_path: str, segs_path: str, ignore_non_match: bool = True)-> List[Tuple[str, str]]:
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    image_files = []
    seg_files = []

    image_files = _get_images_from_dir(images_path)
    seg_files = _get_images_from_dir(segs_path)

    if len(image_files) != len(seg_files):
        raise DataLoaderError(f"Invalid number of image files ({len(image_files)}) vs segment files ({seg_files})")

    fp_pairs = []
    for _img, _seg in zip(image_files, seg_files):
        if _img[0] == _seg[0]:
            fp_pairs.append((_img[2], _seg[2]))

    return fp_pairs


if __name__ == "__main__":
    img_path , seg_path = _get_img_seg_path("dataset/20220125")
    _get_pairs_from_paths(img_path, seg_path)