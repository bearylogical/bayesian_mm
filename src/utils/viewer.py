# from src.utils.loader import get_pairs_from_paths
import os
from pathlib import Path
from re import L

import keras
import matplotlib as mpl
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import cv2
import albumentations as A
import numpy as np

from src.keypoint.predict import predict_imgs_from_dir
from src.processors.imagej import load_imagej_data
from src.utils.dataloader import (
    get_image_paths_from_dir,
    get_img_target_data,
    match_image_to_target,
    get_keypoint_dict_from_ls,
    KeyPointDataLoader,
)
from src.utils.utilities import prepare_img_prediction, get_format_files
from PIL import Image
from typing import Iterable, Union, List, Tuple
import copy
import random
from keras import Model, models


def display_mask(mask):
    """Quick utility to display a model's prediction."""
    # mask = np.argmax(val_preds[i], axis=-1)
    # mask[mask > 0.5] = 1
    # mask[mask <= 0.5] = 0
    # mask = np.expand_dims(mask, axis=-1)
    return tf.keras.preprocessing.image.array_to_img(mask)


def plot_samples_matplotlib(display_list, figsize=(10, 9)):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    fig, axes = plt.subplots(nrows=len(display_list), ncols=3, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i in range(0, 3):
            ax.set_title(title[i % 3])
        if i % 3 != 0 and (i + 1) % 3 == 0:
            t = ax.imshow(display_list[i // 3][i % 3])
            fig.colorbar(t, ax=ax, orientation="vertical")
        else:
            ax.imshow(display_list[i // 3][i % 3], cmap="gray")
    plt.show()


def display_img_annotated(img: Image.Image, data: dict, decimals: int = 3):
    # TODO: might need to make this more generalizable. I.e. for unknown images with supplied types.
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", aspect="auto")
    img_height = img.height
    font_size = 10
    offset = font_size + 3  # kinda arbitrary but it works
    y_pos = img_height - 5  # same here
    text_kwargs = dict(ha="left", va="center", rotation=0, size=font_size)
    for field, prop in data.items():
        ax.text(5, y_pos, f"{field}: {round(prop, decimals)}", **text_kwargs)
        y_pos -= offset
    ax.axis("off")
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
    dpi = float(mpl.rcParams["figure.dpi"])
    return height / dpi, width / dpi


def get_bb_box_outputs(coords: Union[Tuple[int], np.ndarray]):
    if isinstance(coords, np.ndarray):
        assert coords.shape == (4,)

    w = abs(coords[0] - coords[2])
    h = abs(coords[1] - coords[3])

    return coords[:2], w, -h


def display_composite(
    img: np.ndarray,
    bb_box_coords: Union[Tuple[int], np.ndarray],
    coords: Union[np.ndarray, None],
    marker_size: int = 3,
):
    fig = plt.figure(figsize=get_figsize(*img.shape))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin="lower", cmap="gray", aspect="auto")
    bb_min, bb_max = np.amin(bb_box_coords), np.amax(bb_box_coords)
    if bb_max <= 1 and bb_min >= 0:
        x, y, width, height = A.convert_bbox_from_albumentations(
            bb_box_coords, target_format="coco", rows=img.shape[0], cols=img.shape[1]
        )
        xy = (x, y)
    else:
        xy, width, height = get_bb_box_outputs(bb_box_coords)
    rect = Rectangle(xy, width, height, fill=False, linewidth=1, edgecolor="red")

    ax.add_patch(rect)
    coords = coords.reshape(7, 2)
    _plot_keypoints(ax, coords, marker_size, label="Ground Truth")

    plt.axis("off")
    plt.show()


def _plot_keypoints(
    ax: plt.Axes,
    coords: np.ndarray,
    marker_size=3,
    label="Ground Truth",
    marker_fmt="mo",
):
    for idx, (t_x, t_y) in enumerate(coords):
        if idx != 0:
            label = None
        ax.plot(t_x, t_y, marker_fmt, ms=marker_size, label=label)
        ax.annotate(f"{idx}", (t_x, t_y))


def image_j_labels_predict(
    exp_dir: str,
    coord_file: str,
    postfix: str = "_kp_predict",
    model: keras.Model = None,
):
    imgs = get_format_files(
        exp_dir, file_formats=[".tif"], exclude_subdirs=True, sort=True
    )
    x_points, y_points = load_imagej_data(coord_file, normalize=False)
    if model:
        predicted_kps = predict_imgs_from_dir(exp_dir, model, show_predict=False)
    else:
        predicted_kps = None
    for idx, img in enumerate(imgs):
        _img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)

        _img = show_image_coords(
            _img,
            np.array([x_points[idx], y_points[idx]]).T.flatten(),
            predicted_kps[idx] if predicted_kps is not None else None,
            radius=6,
            font_size=25,
        )
        _img.save(img.parent / "predict" / (img.stem + postfix + img.suffix))


def overlay_keypoints(
    img: PIL.Image.Image,
    coords: np.ndarray,
    radius: Union[int, Iterable[int], np.ndarray] = 1,
    show_labels: bool = True,
    color="red",
    xy_offset=(10, -5),
    font_size=6,
):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        # use matplotlib font as backup
        font_path = Path(mpl.get_data_path(), "fonts/ttf/DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), font_size)

    coords = coords.reshape(-1, 2)

    if isinstance(radius, int):
        radius = np.repeat(radius, len(coords) * 2)

    if isinstance(radius, np.ndarray):
        radius = radius.reshape(-1, 2)
        radius.astype(int)

    for idx, (t_x, t_y) in enumerate(coords):
        if show_labels:
            draw.text(
                (t_x + xy_offset[0], t_y + xy_offset[1]),
                text=f"{idx}",
                fill=color,
                font=font,
            )
        draw.ellipse(
            [
                (t_x - radius[idx, 0], t_y - radius[idx, 1]),
                (t_x + radius[idx, 0], t_y + radius[idx, 1]),
            ],
            outline=color,
            fill=color,
        )


def display_img_coords(
    img: np.ndarray,
    true_coords: Union[np.ndarray, None] = None,
    pred_coords: Union[np.ndarray, None] = None,
    marker_size=2,
    true_marker_color="red",
    pred_marker_color="green",
):
    """
    Viewer for overlaying target coords over src image

    :param pred_marker_color: marker color for the predictions. Defaults to green.
    :param true_marker_color: marker color for the ground truth. Defaults to red.
    :param marker_size: Size of markers.
    :param pred_coords: Numpy array of prediction coordinates in [x0, y0, ..,  xN, yN] format
    :param true_coords: Numpy array of prediction coordinates in [x0, y0, ..,  xN, yN] format
    :param img: Image as a Numpy array.
    :return:
    """

    if true_coords is None and pred_coords is None:
        raise Exception("No coords supplied!")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", aspect="auto")

    if true_coords is not None:

        _plot_keypoints(ax, true_coords, marker_size, "Ground Truth")

    if pred_coords is not None:
        _plot_keypoints(ax, pred_coords, marker_size, "Predicted", marker_fmt="cx")

    plt.legend()
    plt.axis("off")
    plt.show()


def show_image_coords(
    img: np.ndarray,
    true_coords: Union[np.ndarray, None] = None,
    pred_coords: Union[np.ndarray, None] = None,
    radius=2,
    true_marker_color="red",
    pred_marker_color="green",
    xy_offset=(10, -5),
    show_labels: bool = True,
    font_size=6,
) -> Image.Image:
    """
    Viewer for overlaying target coords over src image

    :param pred_marker_color: marker color for the predictions. Defaults to green.
    :param true_marker_color: marker color for the ground truth. Defaults to red.
    :param radius: Size of markers.
    :param pred_coords: Numpy array of prediction coordinates in [x0, y0, ..,  xN, yN] format
    :param true_coords: Numpy array of prediction coordinates in [x0, y0, ..,  xN, yN] format
    :param img: Image as a Numpy array.
    :param xy_offset: Offset of keypoint in xy tuple.
    :param show_labels: Flag to display the keypoint label as a text.
    :return:

    Parameters
    ----------
    show_labels
    xy_offset
    xy_offset
    xy_offset
    xy_offset
    """

    if true_coords is None and pred_coords is None:
        raise Exception("No coords supplied!")
    if img.ndim != 3:
        img = Image.fromarray(img).convert("RGB")
    elif img.ndim == 3:
        img = Image.fromarray(img, mode="RGB")
    else:
        raise Exception("Invalid image dimension")

    img_kwargs = {
        "radius": radius,
        "xy_offset": xy_offset,
        "show_labels": show_labels,
        "font_size": font_size,
    }

    if true_coords is not None:
        overlay_keypoints(img, true_coords, color=true_marker_color, **img_kwargs)

    if pred_coords is not None:
        overlay_keypoints(img, pred_coords, color=pred_marker_color, **img_kwargs)

    return img


def display_augmentations(
    dataset: KeyPointDataLoader, batch_idx=0, idx=0, samples=10, cols=5
):
    dataset_copy = copy.deepcopy(dataset)

    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        if i == 0:
            dataset_copy.transform = None
        else:
            dataset_copy.transform = A.Compose(
                [t for t in dataset.transform if not isinstance(t, A.Normalize)],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False, angle_in_degrees=True
                ),
            )
        trans_image, trans_y = dataset_copy[batch_idx][:]
        trans_y = trans_y[idx].reshape(7, 2)
        if dataset.normalize:
            trans_y = trans_y * np.repeat(dataset.img_size, 7).reshape(-1, 2)
        ax.ravel()[i].imshow(
            np.squeeze(trans_image[idx]), origin="lower", cmap="gray", aspect="auto"
        )
        _plot_keypoints(ax.ravel()[i], trans_y)
        ax.ravel()[i].legend()
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def display_keypoints_prediction(
    model: Model, img_path: str, true_keypoints: np.ndarray = None
):
    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pred_coords = model.predict(prepare_img_prediction(img_arr))
    rescale_pred_coords = np.array(
        list(get_keypoint_dict_from_ls(img_arr.shape, list(pred_coords[0])).values())
    )
    # TODO: Refactor as ImageModel method
    display_img_coords(img_arr, true_keypoints, rescale_pred_coords)


def display_predictions(
    model: Model,
    img_dir: Union[str, Path],
    num_images: int = 10,
    rows: int = 2,
    img_size: Tuple[int, int] = (224, 224),
    target_size: Union[Tuple[int, int], None] = (400, 400),
):
    if not isinstance(model, Model):
        raise Exception(f"Incorrect object type of model ({type(model)})")

    if isinstance(img_dir, str):
        img_dir = Path(img_dir)

    cols = num_images // rows
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))

    data_path = img_dir / "targets.npz"
    imgs = get_image_paths_from_dir(img_dir)
    # get random selection of imgs
    rand_imgs = random.sample(list(imgs), num_images)

    for i in range(num_images):
        img, coords = get_img_target_data(rand_imgs[i], data_path, img_size)
        true_coords = np.array([v for v in coords.values()]).reshape(7, 2)
        pred_coords = model.predict(prepare_img_prediction(img)).reshape(
            7, 2
        ) * np.repeat(img_size, 7).reshape(-1, 2)

        if target_size is not None:
            img = cv2.resize(img, target_size)
            target_scale = target_size[0] / img_size[0]
            true_coords = true_coords * target_scale
            pred_coords = pred_coords * target_scale

        ax.ravel()[i].imshow(img, origin="lower", cmap="gray", aspect="auto")
        _plot_keypoints(ax.ravel()[i], true_coords, label="Ground Truth")
        _plot_keypoints(ax.ravel()[i], pred_coords, label="Predicted", marker_fmt="cx")
        ax.ravel()[i].legend()
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
    # img_dir = "dataset/experiments/15Apr"
    # labels_dir = "dataset/experiments/15Apr/labels"
    # # match data and labels
    # imgs, labels = match_image_to_target(img_dir, labels_dir, target_fmt=[".json"])
    # img_path = Path(img_dir) / "8777d5e0-Inc_press_1_seq0023.png"
    # t_img, data = get_img_target_data(Path(img_dir) / "8777d5e0-Inc_press_1_seq0023.png",
    #                                   Path(labels_dir) / "8777d5e0-Inc_press_1_seq0023.json")
    #
    # t_coords = np.array([v for v in data.values()])
    # # vals = np.array([[ 7.471316, 55.25817 , 31.497335, 51.900547, 43.932575, 49.97762 ,
    # #     31.742086, 63.395126, 42.726536, 65.51687 , 29.972242, 57.527832,
    # #     45.319202, 57.237682]])
    # # vals = np.array(list(get_keypoint_from_ls(t_img.shape, vals[0]).values()))
    # show_image_coords(t_img, t_coords).show()
    model = keras.models.load_model("artifacts/trained_model_mc_dropout:v8")
    image_j_labels_predict(
        "dataset/Inc_press_2", "dataset/Inc_press_2/Inc_press_2.txt", model=model
    )
    # model_path = "models/Baseline_20220417_1725"
    # img_model = models.load_model(model_path)
    # display_keypoints_prediction(img_model, str(img_path), t_coords)
