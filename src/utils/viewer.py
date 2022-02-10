# from src.utils.loader import get_pairs_from_paths
import os.path

from PIL.ImageOps import autocontrast
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from src.utils.loader import get_img_target_data
from PIL import Image
from typing import Union, List




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


def display_img_annotated(img: Image.Image, data:dict, decimals:int= 3):
    # TODO: might need to make this more generalizable. I.e. for unknown images with supplied types.
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', aspect='auto')
    img_height = img.height
    font_size = 10
    offset = font_size + 3 # kinda arbitrary but it works
    y_pos = img_height - 5 # same here
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




if __name__ == "__main__":
    # display_img_annotated()
    pass