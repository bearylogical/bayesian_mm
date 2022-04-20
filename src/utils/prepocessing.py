import copy
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
from skimage.draw import line
from src.utils.mechanics import CapillaryStressBalance
from src.utils.geometry import Point
from src.utils.shapes.capillary import sort_xy
from src.utils.transforms import reject_outliers, normalize, divide_by_zero
import random
from pathlib import Path
from time import strftime
from typing import List, Tuple, Union
from PIL import Image

def countour_thresh(img, val=10):
    threshold = val
    # Detect edges using Canny
    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(len(contours)):
        color = random.randint(0, 256)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    plt.imshow(drawing, cmap='gray')
    plt.show()


def _plt_image(img):
    plt.imshow(img, cmap='gray_r')
    plt.show()

def show_lines(img, lines):
    for l in lines:
        draw_line(img, l)
    Image.fromarray(img).show()

def preprocess_lines(img: np.ndarray,
                     debug: bool = False,
                     show_original: bool = False,
                     ratio=3,
                     low_threshold=100):
    # thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    canny_ = cv2.Canny(img, low_threshold, ratio * low_threshold, 3)
    # canny_ = thresh1
    linesP = cv2.HoughLinesP(canny_, 2, np.pi / 180, 40, None, 150, 50)
    # countour_thresh(canny_, 50)
    # _plt_image(canny_)
    img_copy = np.copy(canny_)

    if linesP is not None and len(linesP) > 4:
        print(f'{linesP.shape[0]} Lines Detected')
        linesP = np.squeeze(linesP, 1)
        if debug:
            show_lines(img, linesP)

        try:
            outer_lines = get_outer_edges(linesP)

            _, y_c = get_intersection_point(outer_lines[0], outer_lines[1])

            lb = left_boundary(canny_, midpoint_y=y_c)

            inner_lines = get_inner_edges(linesP, y_c)
            # shift new mid point line to between inner edges

            filtered_lines = np.vstack([outer_lines, inner_lines])
            filtered_lines = np.apply_along_axis(extend_line, axis=1, arr=filtered_lines, xcoord=lb)
            if show_original:
                img_copy = img

                    # cv2.line(img_copy, (l[0], l[1]), (l[2], l[3]), 255, 3)

            for l in filtered_lines:
                # l = extend_line(l_o, lb)
                draw_line(img_copy, l)

            base_y_c = np.min(filtered_lines[2, 1])
            y_c = base_y_c + abs(filtered_lines[2, 1] - filtered_lines[3, 1]) // 2
            mid_line = np.array([lb, y_c, img.shape[1], y_c])
            draw_line(img_copy, mid_line)
            # cv2.line(img_copy,(mid_line[0], mid_line[1]), (mid_line[2], mid_line[3]) , 255, 3)
            cv2.circle(img_copy, (lb, y_c), radius=2, color=255, thickness=-1)

            return img_copy, CapillaryStruct(filtered_lines[2],
                                             filtered_lines[3],
                                             mid_line,
                                             lb=lb)
        except ValueError:
            show_lines(img, linesP)
            exit()
            # cv2.line(img_copy, (l[0], l[1]), (l[2], l[3]), 255, 3)
    else:
        print("No lines detected")

        return img_copy, None


def draw_capillary_outline(img_arr: np.ndarray, params: dict) -> np.ndarray:
    _, w = img_arr.shape
    lb = params["lb"]

    for k, param in params.items():
        if "arr" in k:
            _line_arr = line_start_end_from_params(param[0], param[1], lb, w)
            draw_line(img_arr, _line_arr)

    return img_arr


def line_start_end_from_params(slope: float,
                               intercept: float,
                               x_start: float, x_end: float) -> np.ndarray:
    return np.array([x_start, slope * x_start + intercept,
                     x_end, slope * x_end + intercept], dtype=int)


def draw_line(img_arr: np.ndarray,
              line_arr: np.ndarray,
              line_color: int = 255,
              line_thickness: int = 3):
    if len(line_arr) != 4:
        raise Exception(f"Number of params to define a line is insufficient."
                        f" Expect 4 params, received{line_arr}")

    return cv2.line(img_arr, (line_arr[0], line_arr[1]), (line_arr[2], line_arr[3]), line_color, line_thickness)


class CapillaryStruct:

    def __init__(self, arr_1, arr_2, arr_mid, lb):
        self.arr_1 = arr_1
        self.arr_2 = arr_2
        self.arr_mid = arr_mid
        self.lb = lb
        self.params: dict = dict()
        self.get_params()

    def get_params(self):
        self.params["arr_1"] = get_line_params(self.arr_1)
        self.params["arr_2"] = get_line_params(self.arr_2)
        self.params["arr_mid"] = get_line_params(self.arr_mid)
        self.params["lb"] = self.lb


def get_line_intensity(img_arr: np.ndarray,
                       line_coords: Tuple[np.ndarray],
                       start: Union[None, float] = None,
                       end: Union[None, float] = None,
                       is_normalize: bool = True):
    _intensity = np.zeros(shape=len(line_coords[0]))

    if start is None:
        start = 0
    else:
        start = int(start * len(_intensity))

    if end is None:
        end = -1
    else:
        end = 1 - int(end * len(_intensity))

    for idx, (r, c) in enumerate(zip(*line_coords)):
        try:
            _intensity[idx] = img_arr[r, c]
        except IndexError:
            pass

    _intensity = _intensity[start:end]

    if is_normalize:
        _intensity = normalize(_intensity)

    return _intensity


def get_edge_threshold():
    pass


def plot_line_intensity(img_arr: np.ndarray,
                        line_arr: np.ndarray):
    _img_arr = copy.deepcopy(img_arr)
    line_coords = line(line_arr[1], line_arr[0], line_arr[3], line_arr[2])
    intensities = get_line_intensity(_img_arr,
                                     line_coords,
                                     start=0.01,
                                     end=0.01)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(6, 4))
    _img_arr = draw_line(_img_arr, line_arr)
    ax1.set_title("Raw Image with line overlay")
    ax1.imshow(_img_arr, cmap='gray')
    ax2.set_title("Intensity vs Position")
    ax2.set_ylabel("Intensity (Normalized)")
    ax2.set_xlabel("Position")
    ax2.plot(range(len(intensities)), intensities)

    plt.tight_layout()
    plt.show()
    # return 1


def get_region_of_interest(img_arr: np.ndarray,
                           params: dict):
    _, w = img_arr.shape
    lb = params["lb"]
    #
    # find top line
    _line_arr_1 = line_start_end_from_params(params["arr_1"][0], params["arr_1"][1], lb, w)
    _line_arr_1_coords = np.array(line(_line_arr_1[1], _line_arr_1[0], _line_arr_1[3], _line_arr_1[2]))
    # find bottom line
    _line_arr_2 = line_start_end_from_params(params["arr_2"][0], params["arr_2"][1], lb, w)
    _line_arr_2_coords = np.array(line(_line_arr_2[1], _line_arr_2[0], _line_arr_2[3], _line_arr_2[2]))

    points = np.hstack([_line_arr_1_coords, _line_arr_2_coords]).T
    # points
    # for k, param in params.items():
    #     if "arr" in k:
    #         _line_arr = line_start_end_from_params(param[0], param[1], lb, w)
    #         _line_coords = line(_line_arr[1], _line_arr[0], _line_arr[3], _line_arr[2])
    #         points.extend(_line_coords)
    sorted_points = sort_xy(points[:, 1], points[:, 0])
    return np.array(sorted_points).T


def isolate_particle(img_arr: np.ndarray,
                     points: np.ndarray,
                     dst_img_size: Tuple[int, int] = None,
                     min_threshold: int = 50,
                     ) -> np.ndarray:
    """

    :param dst_img_size: Size of img if specified
    :param img_arr:
    :param points:
    :param min_threshold:
    :return: Numpy image array
    """
    if points.ndim != 2:
        raise Exception("Incorrect dimension")

    _img_arr = copy.deepcopy(img_arr)
    mask = np.zeros(_img_arr.shape[:2], dtype="uint8")
    white_bg = np.full(mask.shape, 255, dtype="uint8")
    cv2.fillPoly(mask, [points], color=255)
    mask_inv = cv2.bitwise_not(mask)

    white_bg = cv2.bitwise_or(white_bg, white_bg, mask=mask_inv)
    masked = cv2.bitwise_and(_img_arr, _img_arr, mask=mask)
    _new_img = white_bg + masked
    _new_img[_new_img < min_threshold] = 255
    if dst_img_size:
        _new_img = cv2.resize(_new_img, dst_img_size)

    return _new_img


def get_auto_contrast_params(img: np.ndarray,
                             percentile: Tuple[int, int] = (0, 95)) -> Tuple[float, float]:
    if percentile[0] > percentile[1]:
        raise Exception("Percentile follows format: (lower, upper)")

    img_hist, _, _, _ = cumfreq(img.ravel(), 256, [0, 256])

    # get 5th percentile gray value
    p_lower = np.argwhere(img_hist > np.percentile(img_hist, percentile[0]))[0].item()
    # get 95th percentile gray value
    p_upper = np.argwhere(img_hist > np.percentile(img_hist, percentile[1]))[0].item()

    # solve ax=b to get phi and gamma
    # we want things at p5 --> 0 and
    # p95 --> 255

    a = np.array([[p_lower, 1], [p_upper, 1]])
    b = np.array([0, 255])

    return tuple(np.linalg.solve(a, b).tolist())


def apply_contrast(img: np.ndarray,
                   method: str = 'auto') -> np.ndarray:
    """
    Apply contrast to img according to transformation
    g' = alpha * g + beta

    :param img:
    :param method: Contrast application. Available methods:
    'auto' - Clips the intensity for the 5 and 70th percentile of cumulative intensity counts
    'clahe' - Applied the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm for region
    based contrast
    :return:
    """
    method = method.lower()
    _img = copy.deepcopy(img)
    if method == 'auto':
        a, b = get_auto_contrast_params(_img, (5, 70))
        return cv2.convertScaleAbs(_img, alpha=a, beta=b)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        return clahe.apply(_img)
    else:
        raise Exception(f'Method {method} unknown. Available methods: auto and clahe.')


def get_mean_params(cap_structs: List[CapillaryStruct]) -> dict:
    """
    Gets the mean params from a list of capillary structs

    :param cap_structs: List of capillary structs
    :return: a dict containing the mean params
    """

    params = dict(arr_1=None, arr_2=None, arr_mid=None, lb=None)
    for k in params.keys():
        _params = np.array([cap.params[k] for cap in cap_structs])
        _params = reject_outliers(_params)
        params[k] = np.mean(_params, axis=0)
    return params


def extend_line(line: np.ndarray, xcoord: int) -> np.ndarray:
    """
    Extend the line up to and including a denoted xcoord
    :param line:
    :param xcoord:
    :return:
    """
    m, b = get_line_params(line)
    start_y = m * xcoord + b
    return np.hstack([xcoord, int(start_y), line[2:]])


def left_boundary(img: np.ndarray, midpoint_y: int):
    """
    Finds the x coordinate of the left boundary
    :param img: image as a Numpy array
    :param midpoint_y: y coordinate of mid-line
    :return:
    """

    for yi in range((img.shape[1])):
        if img[midpoint_y, yi] == 255:
            return yi


def get_outer_edges(lines: np.ndarray):
    """
    Get outer edges simply by looking at the y-coord extrema on either side
    :param img:
    :return:
    """
    ind_min, ind_max = np.argmin(lines[:, 3]), np.argmax(lines[:, 3])
    return lines[[ind_min, ind_max], :]


def get_inner_edges(lines: np.ndarray,
                    y_midpoint: int):
    """
    Get inner edges characterised by closest distance normal to midpoint line
    :param lines:
    :return:
    """
    # above midline
    above_midline = lines[(lines[:, 1] < y_midpoint) & (lines[:, 3] < y_midpoint)]
    above_midline_distance = np.abs(above_midline - y_midpoint)

    # below midline
    below_midline = lines[(lines[:, 3] > y_midpoint)]
    below_midline_distance = np.abs(below_midline - y_midpoint)

    ind_above, ind_below = np.argmin(above_midline_distance[:, 1]), np.argmin(below_midline_distance[:, 1])
    return np.vstack((above_midline[ind_above], below_midline[ind_below]))


def get_intersection_point(line_1: np.ndarray, line_2: np.ndarray):
    line_1_a, line_1_b = get_line_params(line_1)
    line_2_a, line_2_b = get_line_params(line_2)
    a = np.array([[line_1_a, 1.0], [line_2_a, 1.0]])
    b = np.array([line_1_b, line_2_b])

    return np.linalg.solve(a, b).astype(int)


def auto_align(img: np.ndarray):
    pass


def get_line_params(line: np.ndarray):
    if len(line) == 0:
        raise Exception("Line has no data")
    start = Point(*line[:2])
    end = Point(*line[2:])
    slope = get_slope(start, end)
    intercept = get_intercept(slope, start)

    return slope, intercept


def get_slope(start: Point, end: Point):
    return (start.y - end.y) / (start.x - end.x)


def get_intercept(slope: float, point: Point):
    return point.y - slope * point.x


def plot_isolated_particle(img_arr: np.ndarray,
                           params: dict,
                           annotate: bool = True):
    _img_arr = copy.deepcopy(img_arr)

    points = get_region_of_interest(_img_arr, params)
    _img_arr = isolate_particle(_img_arr, points)
    _img_arr = apply_contrast(_img_arr, method='auto')

    # _img_arr = cv2.Canny(_img_arr, 20, 300)
    plt.imshow(_img_arr, cmap='gray')

    if annotate:
        alpha = CapillaryStressBalance.calc_alpha(params["arr_1"][1],
                                                  params["arr_2"][1])
        plt.text(10, 50, f"alpha: {alpha}")

    plt.show()


def save_and_apply_particle_isolation(img_list: List[Union[str, np.ndarray]],
                                      params: dict,
                                      save: bool = False,
                                      save_dir: Union[str, Path, None] = None,
                                      contrast_method='auto'):
    img_list = []
    _img = None
    if save:
        if save_dir is None:
            save_dir = Path(os.getcwd()) / 'dataset' / 'processed' / strftime("%Y%m%d")

        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        if isinstance(img, np.ndarray):
            _img = img
        elif isinstance(img, str):
            _img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        # if _img is not None:
        #     t_dim = _img.shape
        # else:
        #     t_dim = (400, 400)
        points = get_region_of_interest(_img, params)
        _img = isolate_particle(_img, points)
        _img = apply_contrast(_img)

        if save:
            fp = save_dir / Path(img).parts[-1]
            cv2.imwrite(fp, _img)

        img_list.append(_img)

    return img_list


def get_sequence_params(img_list: List[Union[str, np.ndarray, Path]]) -> dict:
    """
    Get relevant parameters from a sequence of images. The sequence should be
    a list of image paths taken within one experiment.

    :param imgs:
    :param img_paths:
    :return: a dictionary containing the (mean) parameters to be used to isolate the capillary

    Parameters
    ----------
    img_list
    """

    cap_structs = []

    for img in img_list:
        if isinstance(img, np.ndarray):
            _img = img
        elif isinstance(img, (Path, str)):
            _img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        else:
            raise Exception('Invalid file type')

        _, cap_struct = preprocess_lines(_img, show_original=True)
        if cap_struct is not None:
            cap_structs.append(cap_struct)

    mean_params = get_mean_params(cap_structs)

    return mean_params


if __name__ == "__main__":
    img_path = Path('dataset/StyleTransfer')
    from src.utils.loader import get_image_paths_from_dir

    imgs = get_image_paths_from_dir(img_path)

    # sample_img = img_path / '006.png'
    t_img_paths = imgs[:6]
    mean_params = get_sequence_params(t_img_paths)
    # save_and_apply_particle_isolation(t_img_paths, mean_params, save=True)

    # fig, ax = plt.subplots(3, 3, figsize=(8, 6))
    # for idx, img_path in enumerate(img_paths):
    #     temp_ax = ax.ravel()[idx]
    #     _img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    #     _img = draw_capillary_outline(_img, mean_params)
    #     temp_ax.imshow(_img, cmap='gray')
    #     temp_ax.set_xticklabels([])
    #     temp_ax.set_yticklabels([])
    #     temp_ax.set_aspect('equal')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.clf()

    t_img = cv2.imread(str(t_img_paths[2]), cv2.IMREAD_GRAYSCALE)
    plt.imshow(t_img, cmap='gray')
    plt.show()
    param = mean_params["arr_1"]
    lb = mean_params["lb"]
    _, w = t_img.shape
    line_arr = line_start_end_from_params(param[0], param[1], lb, w)
    # plot_line_intensity(t_img, line_arr)
    plot_isolated_particle(t_img, mean_params)

    # fig, ax = plt.subplots(3, 3, figsize=(8, 6))
    # # synthesized images , 60 low, ratio3
    # # lab images,
    # for idx, img_path in enumerate(img_paths):
    #     temp_ax = ax.ravel()[idx]
    #     t_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    #     temp_ax.imshow(preprocess_lines(t_img, show_original=False, debug=False)[0], cmap='gray')
    #     temp_ax.set_xticklabels([])
    #     temp_ax.set_yticklabels([])
    #     temp_ax.set_aspect('equal')
    # # fig.subplots_adjust(wspace=0, hspace=0)
    # # t_img = cv2.imread(str(imgs[35]), cv2.IMREAD_GRAYSCALE)
    # # plt.imshow(overlay_lines(t_img, show_original=True, debug=False), cmap='gray_r')
