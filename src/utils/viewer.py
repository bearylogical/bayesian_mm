# from src.utils.loader import get_pairs_from_paths
from PIL.ImageOps import autocontrast
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image



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


if __name__ == "__main__":
    pass
