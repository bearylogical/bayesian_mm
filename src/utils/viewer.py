# from src.utils.loader import get_pairs_from_paths
from pathlib import Path

from PIL.ImageOps import autocontrast
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import tensorflow as tf
import cv2
import albumentations as A
import numpy as np
from src.utils.loader import get_image_paths_from_dir, RegressionDataLoaderT1
from PIL import Image
from typing import Union, List, Tuple
import copy

def display_mask(mask):
    """Quick utility to display a model's prediction."""
    # mask = np.argmax(val_preds[i], axis=-1)
    # mask[mask > 0.5] = 1
    # mask[mask <= 0.5] = 0
    # mask = np.expand_dims(mask, axis=-1)
    return tf.keras.preprocessing.image.array_to_img(mask)


def plot_samples_matplotlib(display_list, figsize=(10, 9)):
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    fig, axes = plt.subplots(nrows=len(display_list), ncols=3, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i in range(0, 3):
            ax.set_title(title[i % 3])
        if i % 3 != 0 and (i + 1) % 3 == 0:
            t = ax.imshow(display_list[i // 3][i % 3])
            fig.colorbar(t, ax=ax, orientation='vertical')
        else:
            ax.imshow(display_list[i // 3][i % 3], cmap='gray')
    plt.show()


def display_img_annotated(img: Image.Image, data: dict, decimals: int = 3):
    # TODO: might need to make this more generalizable. I.e. for unknown images with supplied types.
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', aspect='auto')
    img_height = img.height
    font_size = 10
    offset = font_size + 3  # kinda arbitrary but it works
    y_pos = img_height - 5  # same here
    text_kwargs = dict(ha="left", va="center", rotation=0, size=font_size)
    for field, prop in data.items():
        ax.text(
            5, y_pos, f"{field}: {round(prop, decimals)}", **text_kwargs)
        y_pos -= offset
    ax.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plt.tight_layout(pad=0)

    plt.show()


def get_figsize(height, width):
    """
    Get figsize exactly.
    :param height:
    :param width:
    :return:
    """
    dpi = float(mpl.rcParams['figure.dpi'])
    return height / dpi, width / dpi


def get_bb_box_outputs(coords: Union[Tuple[int], np.ndarray]):
    if isinstance(coords, np.ndarray):
        assert coords.shape == (4,)

    w = abs(coords[0] - coords[2])
    h = abs(coords[1] - coords[3])

    return coords[:2], w, -h


def display_composite(img: Image.Image,
                      bb_box_coords: Union[Tuple[int], np.ndarray],
                      coords: Union[np.ndarray, None],
                      marker_size: int = 3):
    fig = plt.figure(figsize=get_figsize(*img.size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin='lower', cmap='gray', aspect='auto')
    xy, width, height = get_bb_box_outputs(bb_box_coords)
    rect = Rectangle(xy, width, height, fill=False, linewidth=1, edgecolor='red')

    ax.add_patch(rect)
    coords = coords.reshape(7, 2)
    ax.plot(coords[:, 0], coords[:, 1], 'mo', ms=marker_size)
    plt.axis('off')
    plt.show()


def display_img_coords(img: Image.Image,
                       true_coords: Union[np.ndarray, None] = None,
                       pred_coords: Union[np.ndarray, None] = None,
                       marker_size=2,
                       true_marker_color='red',
                       pred_marker_color='green'):
    """
    Viewer for overlaying target coords over src image
    :param pred_marker_color:
    :param true_marker_color:
    :param marker_size:
    :param pred_coords:
    :param true_coords:
    :param img:
    :return:
    """

    true_coords = true_coords.reshape(7, 2)

    if true_coords is None and pred_coords is None:
        raise Exception('No coords supplied!')

    fig = plt.figure(figsize=get_figsize(*img.size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin='lower', cmap='gray', aspect='auto')
    ax.plot(true_coords[:, 0], true_coords[:, 1], 'mo', ms=marker_size, label='Ground Truth')

    if pred_coords is not None:
        pred_coords = pred_coords.reshape(7, 2)
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], 'cx', ms=marker_size, label='Predicted')

    plt.legend()
    plt.axis('off')
    plt.show()


def display_augmentations(dataset: RegressionDataLoaderT1, batch_idx=0, idx=0, samples=10, cols=5):

    if not isinstance(dataset, RegressionDataLoaderT1):
        raise Exception(f'Incorrect object type of dataset ({type(dataset)})')

    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, A.Normalize)],
                                  keypoint_params=A.KeypointParams(format='xy',
                                                                   remove_invisible=False,
                                                                   angle_in_degrees=True))
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        trans_image, trans_y = dataset[batch_idx][:]
        trans_y = trans_y[idx].reshape(7, 2)
        ax.ravel()[i].imshow(trans_image[idx], cmap='gray', aspect='auto')
        ax.ravel()[i].plot(trans_y[:, 0], trans_y[:, 1], 'mo', ms=2, label='Ground Truth')
        ax.ravel()[i].legend()
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # from pathlib import Path
    #
    # img_path = Path('dataset/20220228/images/train')
    # data_path = img_path / 'targets.npz'
    # sample_img = img_path / '1.png'
    #
    # from src.utils.loader import get_img_target_data
    #
    # t_img, data = get_img_target_data(sample_img, data_path, task="T1")
    # _, bb_data = get_img_target_data(sample_img, data_path, task="T2")
    # t_coords = np.array([v for v in data.values()])
    # bb_data = np.array([v for v in bb_data.values()])
    # # display_composite(t_img, bb_box_coords=bb_data, coords=t_coords)
    # import numpy as np
    # pred_coords = np.random.randint(-5, 5, size=t_coords.size) + t_coords
    # display_img_coords(t_img, t_coords, None)

    img_dir = 'dataset/20220228/images/train'
    img_paths = get_image_paths_from_dir(img_dir)
    target_path = 'dataset/20220228/images/train/targets.npz'

    train_gen = RegressionDataLoaderT1(input_img_paths=img_paths,task='T1', batch_size=2, img_size=(128,128), target_paths=target_path)
    display_augmentations(train_gen)

