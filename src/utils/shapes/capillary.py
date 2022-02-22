import random
from PIL import Image, ImageDraw
from src.utils.shapes.shapes import ImageGenerator, \
    get_bezier_parameters, \
    bezier_curve
from itertools import product, chain
import numpy as np
from skimage.draw import line, circle_perimeter
from typing import Tuple, Union, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class InvalidBoundError(Exception):
    pass


class CapillaryImage:

    def __init__(self, yx_r: Tuple[int, int] = (200, 50),
                 theta: Union[int, float] = 2.5,
                 taper_dist: Union[int, float] = 30,
                 taper_to_c1_dist: Union[int, float] = 120,
                 l_b1: int = 20,
                 l_b2: int = 20,
                 l_band: int = 35,
                 taper_cutoff: int = 120,
                 img_size: Tuple[int, int] = (400, 400),
                 is_deg: bool = True):
        """

        :param yx_r: Coordinates of virtual reference point of theta
        :param theta: Angle in degrees of taper
        :param taper_dist: Distance from virtual point to taper, 0 is when there is no cut
        :param taper_to_c1_dist:
        :param l_b1:
        :param l_b2:
        :param l_band:
        :param taper_cutoff:
        :param img_size:
        :param is_deg:
        """
        self.coords = None
        self.volume = None
        self.r_band = None
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
        self.capillary_line_width = 2

    def _check_bounds(self):
        # check for theta (if deg or rad and convert appropriately)
        if self.is_deg:
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
        # check for l_b1
        if (self.l_b1 < self.l_b_min) | (self.l_b2 < self.l_b_min):
            raise InvalidBoundError(f"L_B1 ({self.l_b1}) or L_B2 ({self.l_b2}) below minimum of {self.l_b_min}")
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
        ind_def = (x_L1 >= b1_x1) & (x_L1 <= b2_x1)
        x_L1_v, y_L1_v = x_L1[ind_def], y_L1[ind_def]
        x_L2_v, y_L2_v = x_L2[ind_def], y_L2[ind_def]
        b1_intersect = b1_xf <= min(x_L1_v)

        x_v = np.concatenate([np.flip(b1_xf), x_L1_v, np.flip(b2_xf), np.flip(x_L2_v)])
        y_v = np.concatenate([np.flip(b1_yf), y_L1_v, np.flip(b2_yf), np.flip(y_L2_v)])

        midpoint_x_polygon = b1_x1 + 0.5 * l_band
        r_band = np.round(np.tan(theta) * midpoint_x_polygon, 3)

        r1 = b1_y2 - b1_y1
        r2 = b2_y2 - b2_y1
        volume = round(_calc_vol(r1, r2, b1_x2, b2_x2, l_band), 2)

        # For reframed prediction task
        x0 = yx_r[::-1]  # it was in yx space, convert to xy
        x1 = (b1_x1, b1_y1)
        x2 = (b2_x1, b2_y1)
        x3 = (b1_x3, b1_y3)
        x4 = (b2_x3, b2_y3)
        x5 = (b1_x2, b1_y2)
        x6 = (b2_x2, b2_y2)

        coords = {
            "L1": (x_L1, y_L1),
            "L2": (x_L2, y_L2),
            "B1": (b1_xf, b1_yf),
            "B2": (b2_xf, b2_yf),
            "EL": (x_v, y_v),
            "B3": (b3_xf, b3_yf),
            "T1": np.array([x0, x1, x2, x3, x4, x5, x6]).flatten(),
            "data":
                dict(r_band=r_band, l_band=l_band, volume=volume)
        }

        return coords

    def generate_image(self, img: Image.Image = None, is_annotate: bool = True):
        """
        Generate image onto supplied axes
        :param img: PIL image object
        :param is_annotate: bool, if True prints key info onto canvas
        :return: Nothing
        """
        assert img is not None

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

        draw = ImageDraw.Draw(img)
        draw.polygon(list(np.ravel(coords_EL, 'F')),
                     fill=int(self.fill_alpha_inner * 255))
        # draw.line(list(np.ravel(coords_EL, 'F')), fill=1, width=1, joint='curve')
        draw.line(list(np.ravel(coords_L1, 'F')), fill=self.fill_line, width=self.capillary_line_width)  # make width va
        draw.line(list(np.ravel(coords_L2, 'F')), fill=self.fill_line, width=self.capillary_line_width)
        draw.line(list(np.ravel(coords_b3, 'F')), fill=self.fill_line, joint='curve', width=self.capillary_line_width)

        # draw.polygon(list(np.ravel(coords_EL,'F')),
        #              fill=None,
        #              outline=int(self.fill_alpha_outer * 255))

        #
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

    def __init__(self, save_dir=None, num_images: int = 1, **kwargs):
        super(CapillaryImageGenerator, self).__init__(save_dir, **kwargs)
        self.num_images = num_images

        # generator params

    def generate(self):
        # available_theta = np.linspace(2, 20, num=self.num_images, dtype=int)

        available_ref_coord = [(200, 50), ]
        available_l_band = np.linspace(1, 20, num=self.num_images, dtype=int)
        available_taper_c1_dist = np.linspace(80, 150, num=self.num_images, dtype=int)

        parameter_space = product(available_ref_coord,
                                  available_l_band,
                                  available_taper_c1_dist)
        selected_params = random.sample(list(parameter_space), self.num_images)
        # initialise our results array : [idx, lband, rband, vol, theta]
        idx_dtype = ('idx', 'i4')
        T0_dtypes = [idx_dtype, ('l_band', 'f4'), ('r_band', 'f4'),
                     ('volume', 'f4'), ('theta', 'f4')]

        T1_dtypes = [('idx', 'i4')]
        T1_coords = [[(f'x{x}', 'i4'), (f'y{x}', 'i4')] for x in range(7)]
        T1_dtypes.extend(list(chain.from_iterable(T1_coords)))

        res_T0 = np.zeros(self.num_images,
                          dtype=T0_dtypes)
        res_T1 = np.zeros(self.num_images, dtype=T1_dtypes)
        for idx, param in enumerate(tqdm(selected_params)):
            capillary = CapillaryImage(yx_r=param[0],
                                       l_band=param[1],
                                       taper_to_c1_dist=param[2])

            temp_image = Image.new(mode='L', size=capillary.dim, color=255)
            capillary.generate_image(temp_image, is_annotate=False)

            img_fp = str(save_dir / str(idx).zfill(len(str(self.num_images)))) + '.png'

            # plt.autoscale(tight=True)
            temp_image.save(img_fp)
            res_T0[idx] = np.array([(idx, capillary.l_band, capillary.r_band, capillary.volume, capillary.theta)],
                                   dtype=T0_dtypes)

            res_T1[idx] = np.array([(idx, *capillary.coords)], dtype=T1_dtypes)

        res_fp = str(save_dir / 'targets')
        np.savez(res_fp, T0=res_T0, T1=res_T1, allow_pickle=False)


def _calc_vol(r1, r2, b1, b2, l_band):
    # split volume into 3 components
    # 2 half-ellipsoid + 1 conical frustum
    vol_e1 = np.divide(2, 3) * np.pi * r1 ** 2 * b1
    vol_e2 = np.divide(2, 3) * np.pi * r2 ** 2 * b2
    vol_cf = np.divide(np.pi * l_band, 3) * (r1 ** 2 + r1 * r2 + r2 ** 2)
    return vol_e1 + vol_e2 + vol_cf


if __name__ == "__main__":
    # single_cap = CapillaryImage()
    # fig, ax = plt.subplots()
    # single_cap.generate_image(ax)
    # plt.show()
    cap = CapillaryImageGenerator(save_dir=None, num_images=100)
    cap.generate()
