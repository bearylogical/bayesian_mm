
# from src.utils.loader import get_pairs_from_paths
from PIL.ImageOps import autocontrast
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def view_masks(img_path):
    return autocontrast(Image.open(img_path))

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

if __name__ == "__main__":
    img = view_masks("dataset/20220125/segment/00003.png")
    img.show()
# def visualize_segmentation_dataset(images_path, segs_path, n_classes,
#                                    do_augment=False, ignore_non_matching=False,
#                                    no_show=False, image_size=None, augment_name="aug_all", custom_aug=None):
#     try:
#         # Get image-segmentation pairs
#         img_seg_pairs = get_pairs_from_paths(
#                             images_path, segs_path,
#                             ignore_non_matching=ignore_non_matching)
#
#         # Get the colors for the classes
#         colors = class_colors
#
#         print("Please press any key to display the next image")
#         for im_fn, seg_fn in img_seg_pairs:
#             img = cv2.imread(im_fn)
#             seg = cv2.imread(seg_fn)
#             print("Found the following classes in the segmentation image:",
#                   np.unique(seg))
#             img, seg_img = _get_colored_segmentation_image(
#                                                     img, seg, colors,
#                                                     n_classes,
#                                                     do_augment=do_augment, augment_name=augment_name, custom_aug=custom_aug)
#
#             if image_size is not None:
#                 img = cv2.resize(img, image_size)
#                 seg_img = cv2.resize(seg_img, image_size)
#
#             print("Please press any key to display the next image")
#             cv2.imshow("img", img)
#             cv2.imshow("seg_img", seg_img)
#             cv2.waitKey()
#     except DataLoaderError as e:
#         print("Found error during data loading\n{0}".format(str(e)))
#         return False
