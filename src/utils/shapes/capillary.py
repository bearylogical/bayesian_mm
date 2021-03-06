import random
from PIL import Image, ImageDraw
from src.utils.shapes.shapes import ImageGenerator, apply_keypoints_scale, get_scale_factor
from src.utils.utilities import get_PIL_version
from itertools import product, chain
import numpy as np
from albumentations.augmentations.bbox_utils import convert_bbox_to_albumentations
from skimage.draw import line, circle_perimeter
from typing import Tuple, Union, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class InvalidBoundError(Exception):
    pass

class GeometryError(Exception):
    pass


class CapillaryImage:

    def __init__(self, yx_r: Tuple[int, int] = (200, 50),
                 theta: Union[int, float] = None,
                 taper_dist: Union[int, float] = 30,
                 taper_to_c1_dist: Union[int, float] = 120,
                 l_b1: int = None,
                 l_b2: int = None,
                 r_band:int = None,
                 l_band: int = None,
                 taper_cutoff: int = 120,
                 img_size: Tuple[int, int] = (400, 400),
                 is_deg: bool = True):
        """

        :param yx_r: Coordinates of virtual reference point of theta
        :param theta: Half angle (in degrees) of taper
        :param taper_dist: Distance from virtual point to taper, 0 is when there is no cut
        :param taper_to_c1_dist:
        :param l_b1:
        :param l_b2:
        :param l_band:
        :param taper_cutoff:
        :param img_size:
        :param is_deg:
        """
        self.bounding_box = None
        self.coords = None
        self.volume = None
        self.r_band = r_band
        self.theta = theta
        self.yx_r = yx_r
        self.taper_dist = taper_dist
        self.taper_to_c1_dist = taper_to_c1_dist
        self.l_b1 = l_b1
        self.l_b2 = l_b2
        self.l_band = l_band
        self.taper_cutoff = taper_cutoff
        self.is_deg = is_deg
        self.dim = img_size

        # arbitrary visual constants
        self.taper_dist_min = 20
        self.taper_to_c1_dist_min = 30
        self.l_b_min = 5
        self.l_band_min = 0  # including
        self.capillary_close_depth = 5

        self._check_bounds()

        # image stuff
        self.figsize = (10, 10)
        self.fill_alpha_inner = 0.9  # alpha (transparency) for the "confined" ellipsoid # 1 is full
        self.fill_alpha_outer = 0.3
        self.fill_line = 1
        self.capillary_line_width = 10

    def _check_bounds(self):
        if self.r_band is None:
            # check for l_b1
            if (self.l_b1 < self.l_b_min) | (self.l_b2 < self.l_b_min):
                raise InvalidBoundError(f"L_B1 ({self.l_b1}) or L_B2 ({self.l_b2}) below minimum of {self.l_b_min}")

        # check for theta (if deg or rad and convert appropriately)
        if self.is_deg and self.theta is not None:
            self.theta = np.deg2rad(self.theta)
        # check for yx_r coord ranges (bounds)
        if (self.yx_r[0] > self.dim[1]) | \
                (self.yx_r[1] > self.dim[0]) | \
                ((self.yx_r[0] * self.yx_r[1]) < 0):
            raise InvalidBoundError(f"Invalid bound {self.yx_r} within {self.dim}")
        # check for taper_dist (bounds)
        if self.taper_dist <= 0 | self.taper_dist <= self.taper_dist_min:
            raise InvalidBoundError(f"Taper dist {self.taper_dist} < {self.taper_dist_min}")
        # check for taper_to_c1_dist (bounds)
        if self.taper_to_c1_dist < self.taper_to_c1_dist_min:
            raise InvalidBoundError(f"Taper C1 dist {self.taper_to_c1_dist} < {self.taper_to_c1_dist_min}")

        # check for l_band
        if self.l_band < self.l_band_min:
            raise InvalidBoundError(f"L_band {self.l_band} below minimum of {self.l_band_min}")
        # check for taper_cutoff

    @staticmethod
    def _generate_capillary(yx_r: Tuple[int, int] = (200, 50),
                            theta: Union[int, float] = 0.0436,
                            taper_dist: Union[int, float] = 30,
                            taper_to_c1_dist: Union[int, float] = 200,
                            l_b1: int = 20,
                            l_b2: int = 20,
                            l_band: int = 35,
                            taper_cutoff: int = 120,
                            capillary_close_depth: int = 5,
                            img_size: Tuple[int, int] = (400, 400), ) -> Dict:
        """

        :rtype: Dict
        """
        y_r, x_r = yx_r
        v1_c1 = taper_dist + taper_to_c1_dist
        ref_image = np.zeros(img_size)

        # create our imaginary capillary
        y_L1, x_L1 = line(y_r, x_r, int(np.tan(theta) * (ref_image.shape[0] - x_r)) + y_r, ref_image.shape[0] - 1)
        y_L2, x_L2 = line(y_r, x_r, y_r - int(np.tan(theta) * (ref_image.shape[0] - x_r)), ref_image.shape[0] - 1)

        # b1, b2 are the curvatures that define our particle interfaces
        # points for b1
        b1_x1 = x_r + v1_c1  # center of circle (x-coord)
        b1_y1 = y_L1[x_L1 == b1_x1][0]
        # b1_x1 = np.divide(r_band, np.tan(theta)) - l_band / 2
        # b1_radius = int(np.tan(theta) * v1_c1)
        b1_radius = b1_y1 - y_r
        # b1_y1 = y_r + b1_radius
        b1_x2 = b1_x1 - b1_radius  # edge of circle (x-coord)
        b1_y2 = y_r  # edge of circle (y-coord)
        b1_x3 = b1_x1
        b1_y3 = y_L2[x_L2 == b1_x1][0]
        # b1_x = [b1_x1, b1_x2, b1_x3]
        # b1_y = [b1_y1, b1_y2, b1_y3]
        # generate circle coods
        b1_ry, b1_cx = circle_perimeter(b1_y2, b1_x1, b1_radius, shape=img_size)
        # indices of half circle
        b1_mask = b1_cx <= b1_x1
        b1_xf, b1_yf = b1_cx[b1_mask], b1_ry[b1_mask]
        # b1_xf, b1_yf = sort_xy(b1_xf, b1_xf)
        # b1_p = get_bezier_parameters(b1_x, b1_y, degree=2)
        # b1_xf, b1_yf = bezier_curve(b1_p, n_times=50)

        # points for b2
        b2_x1 = b1_x1 + l_band
        b2_y1 = y_L1[x_L1 == b2_x1][0]
        # b2_radius = int(np.tan(theta) * (l_band + v1_c1))
        b2_radius = b2_y1 - y_r
        # b2_y1 = y_r + b2_radius
        b2_x2 = b2_x1 + b2_radius
        b2_y2 = y_r
        b2_x3 = b2_x1
        b2_y3 = y_L2[x_L2 == b2_x1][0]
        # b2_x = [b2_x1, b2_x2, b2_x3]
        # b2_y = [b2_y1, b2_y2, b2_y3]
        # generate circle coords
        b2_ry, b2_cx = circle_perimeter(b2_y2, b2_x1, b2_radius, shape=img_size)
        # indices of half circle
        b2_mask = b2_cx >= b2_x1
        b2_xf, b2_yf = b2_cx[b2_mask], b2_ry[b2_mask]
        # b2_xf, b2_yf = sort_xy(b2_xf, b2_yf)
        # b2_p = get_bezier_parameters(b2_x, b2_y, degree=2)
        # b2_xf, b2_yf = bezier_curve(b2_p, n_times=50)

        # some transformations to generate the taper cutoff
        if (taper_cutoff < (taper_dist + x_r)) | (taper_cutoff > b1_x2):
            taper_cutoff = taper_dist + x_r

        ind_f = x_L1 >= taper_cutoff
        y_L1, x_L1 = y_L1[ind_f], x_L1[ind_f]
        y_L2, x_L2 = y_L2[ind_f], x_L2[ind_f]

        # points for taper cutoff curvature
        b3_x1 = x_L1[0]
        b3_y1 = y_L1[0]
        b3_x2 = b3_x1 - capillary_close_depth
        b3_y2 = y_r
        b3_x3 = x_L1[0]
        b3_y3 = y_L2[0]
        b3_x = [b3_x1, b3_x2, b3_x3]
        b3_y = [b3_y1, b3_y2, b3_y3]
        # generate bezier fit
        # b3_p = get_bezier_parameters(b3_x, b3_y, degree=2)
        b3_xf, b3_yf = b3_x, b3_y

        # get geometry of deformed particle
        # need to find intersection of x, y coord with line
        x_ind_def = (x_L1 >= b1_x1) & (x_L1 <= b2_x1)
        y_ind_def = (y_L1 <= b1_y1) & (y_L1 >= b2_x1)
        x_L1_v, y_L1_v = x_L1[x_ind_def], y_L1[x_ind_def]
        x_L2_v, y_L2_v = x_L2[y_ind_def], y_L2[y_ind_def]
        b1_intersect = b1_xf <= min(x_L1_v)

        x_v = np.concatenate([b1_xf, x_L1_v, b2_xf, x_L2_v])
        y_v = np.concatenate([b1_yf, y_L1_v, b2_yf, y_L2_v])

        x_v, y_v = sort_xy(x_v, y_v)

        midpoint_x_polygon = b1_x1 + 0.5 * l_band
        r_band = np.round(np.tan(theta) * midpoint_x_polygon, 3)

        r1 = b1_y2 - b1_y1
        r2 = b2_y2 - b2_y1
        volume = round(_calc_vol(r1, r2, b1_x2, b2_x2, l_band), 2)

        # For reframed prediction task
        x0 = (b3_x1, b3_y1)
        x1 = (b1_x1, b1_y1)
        x2 = (b2_x1, b2_y1)
        x3 = (b1_x3, b1_y3)
        x4 = (b2_x3, b2_y3)
        x5 = (b1_x2, b1_y2)
        x6 = (b2_x2, b2_y2)

        # for bounding box
        x_min, y_min = b1_x2, b2_y1
        x_max, y_max = b2_x2, b2_y3

        coords = {
            "L1": (x_L1, y_L1),
            "L2": (x_L2, y_L2),
            "B1": (b1_xf, b1_yf),
            "B2": (b2_xf, b2_yf),
            "EL": (x_v, y_v),
            "B3": (b3_xf, b3_yf),
            "T1": np.array([x0, x1, x2, x3, x4, x5, x6]).flatten(),
            "T2": np.array([x_min, y_min, x_max, y_max]).flatten(),
            "data":
                dict(r_band=r_band, l_band=l_band, volume=volume)
        }

        return coords

    def _generate_capillary_v2(
            self,
            ) -> Dict:
        """

        :rtype: Dict
        """

        l_d, v1_c1, r_1, r_2 = map(int, decompose_l_r_band(self.theta, self.l_band, self.r_band))
        x_r, y_r = self.yx_r[1], self.dim[1] // 2 # auto centering in the y axis
        ref_image = np.zeros(self.dim)

        # create our imaginary capillary
        y_L1, x_L1 = line(y_r, x_r, int(np.tan(self.theta) * (ref_image.shape[0] - x_r)) + y_r, ref_image.shape[0] - 1)
        y_L2, x_L2 = line(y_r, x_r, y_r - int(np.tan(self.theta) * (ref_image.shape[0] - x_r)), ref_image.shape[0] - 1)

        # b1, b2 are the curvatures that define our particle interfaces
        # points for b1
        b1_x1 = x_r + v1_c1  # center of circle (x-coord)
        b1_y1 = y_L1[x_L1 == b1_x1][0]
        # b1_x1 = np.divide(r_band, np.tan(theta)) - l_band / 2
        # b1_radius = int(np.tan(theta) * v1_c1)
        # b1_y1 = y_r + b1_radius
        b1_x2 = b1_x1 - r_1  # edge of circle (x-coord)
        b1_y2 = y_r  # edge of circle (y-coord)
        b1_x3 = b1_x1
        b1_y3 = y_L2[x_L2 == b1_x1][0]
        # b1_x = [b1_x1, b1_x2, b1_x3]
        # b1_y = [b1_y1, b1_y2, b1_y3]
        # generate circle coods
        b1_ry, b1_cx = circle_perimeter(b1_y2, b1_x1, r_1, shape=self.dim)
        # indices of half circle
        b1_mask = b1_cx <= b1_x1
        b1_xf, b1_yf = b1_cx[b1_mask], b1_ry[b1_mask]
        # b1_xf, b1_yf = sort_xy(b1_xf, b1_xf)
        # b1_p = get_bezier_parameters(b1_x, b1_y, degree=2)
        # b1_xf, b1_yf = bezier_curve(b1_p, n_times=50)f

        # points for b2
        b2_x1 = b1_x1 + self.l_band
        b2_y1 = y_L1[x_L1 == b2_x1][0]
        b2_x2 = b2_x1 + r_2
        b2_y2 = y_r
        b2_x3 = b2_x1
        b2_y3 = y_L2[x_L2 == b2_x1][0]
        # b2_x = [b2_x1, b2_x2, b2_x3]
        # b2_y = [b2_y1, b2_y2, b2_y3]
        # generate circle coords
        b2_ry, b2_cx = circle_perimeter(b2_y2, b2_x1, r_2, shape=self.dim)
        # indices of half circle
        b2_mask = b2_cx >= b2_x1
        b2_xf, b2_yf = b2_cx[b2_mask], b2_ry[b2_mask]
        # b2_xf, b2_yf = sort_xy(b2_xf, b2_yf)
        # b2_p = get_bezier_parameters(b2_x, b2_y, degree=2)
        # b2_xf, b2_yf = bezier_curve(b2_p, n_times=50)

        # some transformations to generate the taper cutoff
        if (self.taper_cutoff < (self.taper_dist + x_r)) | (self.taper_cutoff > b1_x2):
            self.taper_cutoff = self.taper_dist + x_r

        ind_f = x_L1 >= self.taper_cutoff
        y_L1, x_L1 = y_L1[ind_f], x_L1[ind_f]
        y_L2, x_L2 = y_L2[ind_f], x_L2[ind_f]

        # points for taper cutoff curvature
        b3_x1 = x_L1[0]
        b3_y1 = y_L1[0]
        b3_x2 = b3_x1 - self.capillary_close_depth
        b3_y2 = y_r
        b3_x3 = x_L1[0]
        b3_y3 = y_L2[0]
        b3_x = [b3_x1, b3_x2, b3_x3]
        b3_y = [b3_y1, b3_y2, b3_y3]
        # generate bezier fit
        # b3_p = get_bezier_parameters(b3_x, b3_y, degree=2)
        b3_xf, b3_yf = b3_x, b3_y

        # get geometry of deformed particle
        # need to find intersection of x, y coord with line
        x_ind_def = (x_L1 >= b1_x1) & (x_L1 <= b2_x1)
        y_ind_def = (y_L1 <= b1_y1) & (y_L1 >= b2_x1)
        x_L1_v, y_L1_v = x_L1[x_ind_def], y_L1[x_ind_def]
        x_L2_v, y_L2_v = x_L2[y_ind_def], y_L2[y_ind_def]
        b1_intersect = b1_xf <= min(x_L1_v)

        x_v = np.concatenate([b1_xf, x_L1_v, b2_xf, x_L2_v])
        y_v = np.concatenate([b1_yf, y_L1_v, b2_yf, y_L2_v])

        x_v, y_v = sort_xy(x_v, y_v)

        midpoint_x_polygon = b1_x1 + 0.5 * self.l_band
        r_band = np.round(np.tan(self.theta) * midpoint_x_polygon, 3)
        r1 = b1_y2 - b1_y1
        r2 = b2_y2 - b2_y1
        volume = round(_calc_vol(r1, r2, b1_x2, b2_x2, self.l_band), 2)

        # For reframed prediction task
        x0 = (b3_x1, b3_y1)
        x1 = (b1_x1, b1_y1)
        x2 = (b2_x1, b2_y1)
        x3 = (b1_x3, b1_y3)
        x4 = (b2_x3, b2_y3)
        x5 = (b1_x2, b1_y2)
        x6 = (b2_x2, b2_y2)

        # for bounding box
        x_min, y_min = b1_x2, b2_y1
        x_max, y_max = b2_x2, b2_y3

        coords = {
            "L1": (x_L1, y_L1),
            "L2": (x_L2, y_L2),
            "B1": (b1_xf, b1_yf),
            "B2": (b2_xf, b2_yf),
            "EL": (x_v, y_v),
            "B3": (b3_xf, b3_yf),
            "T1": np.array([x0, x1, x2, x3, x4, x5, x6]).flatten(),
            "T2": np.array([x_min, y_min, x_max, y_max]).flatten(),
            "data":
                dict(r_band=self.r_band, l_band=self.l_band, volume=volume)
        }

        return coords

    def generate_image(self, img: Image.Image = None, is_annotate: bool = True, version:int=2):
        """
        Generate image onto supplied axes
        :param img: PIL image object
        :param is_annotate: bool, if True prints key info onto canvas
        :return: Nothing
        """
        assert img is not None

        if version == 2:
            coords = self._generate_capillary_v2()
        else:
            coords = self._generate_capillary(self.yx_r,
                                              self.theta,
                                              self.taper_dist,
                                              self.taper_to_c1_dist,
                                              self.l_b1,
                                              self.l_b2,
                                              self.l_band,
                                              self.taper_cutoff,
                                              self.capillary_close_depth,
                                              self.dim)

        coords_L1 = coords["L1"]
        coords_L2 = coords["L2"]
        coords_b1 = coords["B1"]
        coords_b2 = coords["B2"]
        coords_b3 = coords["B3"]
        coords_EL = coords["EL"]

        # store meta info
        self.l_band = coords["data"]["l_band"]
        self.r_band = coords["data"]["r_band"]
        self.volume = coords["data"]["volume"]
        self.coords = coords["T1"]
        self.bounding_box = coords["T2"]

        draw = ImageDraw.Draw(img)
        draw.polygon(list(np.ravel(coords_EL, 'F')),
                     fill=int(self.fill_alpha_inner * 255),
                     outline=int(self.fill_alpha_outer * 255))
        # x_s, y_s = coords_EL
        # assert len(x_s) == len(y_s)
        #
        # draw.line(list(np.ravel(coords_EL, 'F')), fill=self.fill_line, joint='curve', width=1)
        # draw.line([x_s[-1], y_s[-1], x_s[0], y_s[0]], fill=self.fill_line, joint='curve', width=1)

        # compatibility layer, somehow breaks at PIL 7.
        if int(get_PIL_version()[0]) < 9:
            coords_b3 = [tuple(xy) for xy in np.ravel(coords_b3, 'F').reshape(-1, 2).tolist()]
        else:
            coords_b3 = list(np.ravel(coords_L2, 'F'))

        # draw.line(list(np.ravel(coords_EL, 'F')), fill=1, width=1, joint='curve')
        draw.line(list(np.ravel(coords_L1, 'F')), fill=self.fill_line, width=self.capillary_line_width)  # make width va
        draw.line(list(np.ravel(coords_L2, 'F')), fill=self.fill_line, width=self.capillary_line_width)
        draw.line(coords_b3, fill=self.fill_line, joint='curve', width=self.capillary_line_width)

        if is_annotate:
            # text_kwargs = dict(ha="left", va="center", rotation=0, size=10)
            draw.text(
                (20, self.dim[1] - 10), text=f"L_Band: {self.l_band}")
            draw.text(
                (20, self.dim[1] - 25), text=f"R_Band: {self.r_band}")
            draw.text(
                (20, self.dim[1] - 40), text=f"Vol: {self.volume}")


class CapillaryImageGenerator(ImageGenerator):
    """
    Generator to spit out capillary images
    """

    def __init__(self, save_dir=None,
                 num_images: int = 1,
                 target_size: Tuple[int, int]=(1200, 1200), **kwargs):
        super(CapillaryImageGenerator, self).__init__(save_dir, **kwargs)
        self.num_images = num_images
        self.target_size = target_size

        # generator params
        self._generated_resolution = (1200, 1200)
        self.scale = get_scale_factor(self._generated_resolution, self.target_size)

    def generate(self):
        available_theta = np.unique(np.linspace(2, 6, num=self.num_images, dtype=int))
        available_ref_coord = [(600, 150)]
        available_l_band = np.unique(np.linspace(1,120, num=self.num_images, dtype=int))
        available_taper_c1_dist = np.unique(np.linspace(240,450, num=self.num_images, dtype=int))

        parameter_space = product(available_ref_coord,
                                  available_l_band,
                                  available_taper_c1_dist,
                                  available_theta)
        selected_params = random.sample(list(parameter_space), self.num_images)
        # initialise our results array : [idx, lband, rband, vol, theta]
        if self.is_train_split:
            num_train = int(self.train_test_ratio * self.num_images)
            train_params = selected_params[:num_train]
            test_params = selected_params[num_train:]

            self._generate_image(train_params, self.train_dir)
            self._generate_image(test_params, self.test_dir)

        else:
            self._generate_image(selected_params, self.save_img_dir)

    def generate_sequences(self):

        # lbands should increase as time goes by
        # taper to c1 dist should decrease as time goes by (particle moves closer)
        pass

    def _generate_image(self, selected_params: list, save_dir: Path):
        idx_dtype = ('idx', 'i4')
        T0_dtypes = [idx_dtype, ('l_band', 'f4'), ('r_band', 'f4'),
                     ('volume', 'f4'), ('theta', 'f4')]

        T1_dtypes = [('idx', 'i4')]
        T1_coords = [[(f'x{x}', 'i4'), (f'y{x}', 'i4')] for x in range(7)]
        T1_dtypes.extend(list(chain.from_iterable(T1_coords)))

        T2_dtypes = [idx_dtype, (f'x_min', 'f4'), (f'y_min', 'f4'), (f'x_max', 'f4'), (f'y_max', 'f')]

        res_T0 = np.zeros(len(selected_params),
                          dtype=T0_dtypes)
        res_T1 = np.zeros(len(selected_params), dtype=T1_dtypes)
        res_T2 = np.zeros(len(selected_params), dtype=T2_dtypes)

        for idx, param in enumerate(tqdm(selected_params)):
            capillary = CapillaryImage(yx_r=param[0],
                                       l_band=param[1],
                                       taper_to_c1_dist=param[2],
                                       theta=param[3],
                                       img_size=self._generated_resolution,
                                       taper_dist=300)

            temp_image = Image.new(mode='L', size=capillary.dim, color=255)

            capillary.generate_image(temp_image, is_annotate=False)

            img_fp = str(save_dir / str(idx).zfill(len(str(len(selected_params))))) + '.png'
            # temp_image = temp_image.resize(size=(capillary.dim[0] * 3, capillary.dim[1] * 3), resample=Image.ANTIALIAS)

            # convert to numpy array

            temp_image.thumbnail(size=self.target_size, resample=Image.ANTIALIAS)
            # plt.autoscale(tight=True)
            temp_image.save(img_fp)
            res_T0[idx] = np.array([(idx, capillary.l_band, capillary.r_band, capillary.volume, capillary.theta)],
                                   dtype=T0_dtypes)

            res_T1[idx] = np.array([(idx, *(capillary.coords * np.repeat(self.scale, 7)))], dtype=T1_dtypes)
            res_T2[idx] = np.array([(idx, *convert_bbox_to_albumentations(capillary.bounding_box,
                                                                          'pascal_voc',
                                                                          rows=self._generated_resolution[0],
                                                                          cols=self._generated_resolution[1]))], dtype=T2_dtypes)

        res_fp = str(save_dir / 'targets')
        np.savez(res_fp, T0=res_T0, T1=res_T1, T2=res_T2, allow_pickle=False)

def get_theta(l_band:float, r_band:float)->float:
    """
    Get theta from a specified l_band and r_band

    :param l_band: float
    :param r_band: float
    :return: theta as a float.
    """
    return np.arctan2(l_band / 2, r_band)

def create_image(radius:float, length:float):
    pass

def _calc_vol(r1, r2, b1, b2, l_band):
    # split volume into 3 components
    # 2 half-ellipsoid + 1 conical frustum
    vol_e1 = np.divide(2, 3) * np.pi * r1 ** 2 * b1
    vol_e2 = np.divide(2, 3) * np.pi * r2 ** 2 * b2
    vol_cf = np.divide(np.pi * l_band, 3) * (r1 ** 2 + r1 * r2 + r2 ** 2)
    return vol_e1 + vol_e2 + vol_cf


def sort_xy(x: np.ndarray, y: np.ndarray):
    """
    Sorts x,y vertices of a polygon using the center of mass.

    :param x: Array of x coordinates
    :param y: Array of y coordinates
    :return: tuple of sorted (x,y) coordinates
    """
    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    angles = np.where((y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r))

    mask = np.argsort(angles)

    return x[mask], y[mask]


def decompose_l_r_band(theta, l_band, r_band)->tuple:
    """
    Internal function to decompose the l and r band to get r_a and r_b, l_d and L and l_a
    respectively

    :return: Tuple of (v1_c1, r_a, r_b)
    """
    if r_band is None:
        raise GeometryError('No r_band specified')
    # if self.theta is None: # in subsequent images, we find that theta is fixed.
    #     self.theta = get_theta(l_band=self.l_band, r_band=self.r_band)
    l_d = r_band / np.tan(theta)

    v1_c1 =l_d - l_band / 2 # dist from reference to r_1
    r_a = v1_c1 * (r_band / l_d)
    r_b = (v1_c1 + l_band) * (r_band / l_d)

    return l_d, v1_c1, r_a, r_b

if __name__ == "__main__":
    single_cap = CapillaryImage(theta=3.12 * 2, l_band=70, r_band=25)
    temp_image = Image.new(mode='L', size=single_cap.dim, color=255)
    single_cap.generate_image(temp_image)
    plt.imshow(temp_image,  cmap='gray', aspect='auto')
    plt.show()
    # cap = CapillaryImageGenerator(save_dir=None, num_images=1000, force_images=True, target_size=(400, 400))
    # cap.generate()
    # a = apply_keypoints_scale(((200.0,200.0) ,(200.0, 200.0)), (400, 400), (1200, 1200))