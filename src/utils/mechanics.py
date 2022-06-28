from typing import List, Union, Tuple
from copy import deepcopy

import numpy as np

from src.processors.imagej import load_imagej_data, normalize_keypoint, load_pressure_data, load_keypoints_pressures
from src.utils.constants import BULK_MODULUS
import matplotlib.pyplot as plt

from src.utils.geometry import Point, Line, CapillaryPoints, get_intersection, dist_2D, get_line_param
from src.utils.shapes.capillary import decompose_l_r_band
from src.utils.transforms import divide_by_zero, normalise_bands


# TODO: Unit tests


class CapillaryStressBalance:
    """
    Object to calculate the stresses given data.

    """

    def __init__(self, num_points_per_image: int = 7, normalize: bool = True):
        """
        Instance variables for CapillaryStressBalance .

        Parameters
        ----------
        num_points_per_image : Number of points per image.
        """
        self.x_coords = None
        self.y_coords = None
        self.pressures = None
        self.capillary_points: Union[None, List[CapillaryPoints]] = None
        self.num_images = None

        self.normalize = normalize
        self.num_points_per_image = num_points_per_image

        self.l_bands: Union[np.ndarray, None] = None

        self.r_bands: Union[np.ndarray, None] = None
        self.lengths: Union[np.ndarray, None] = None
        self.volumes: Union[np.ndarray, None] = None
        self.wall_pressures: Union[np.ndarray, None] = None
        self.wall_min_pressures: Union[np.ndarray, None] = None
        self.avg_pressures: Union[np.ndarray, None] = None
        self.areas: Union[np.ndarray, None] = None

        self.v_strain: Union[np.ndarray, None] = None
        self.eps_z: Union[np.ndarray, None] = None
        self.eps_r: Union[np.ndarray, None] = None
        self.eps_g: Union[np.ndarray, None] = None
        self.K_compressive: Union[np.ndarray, None] = None
        self.G_shear: Union[np.ndarray, None] = None

        self.alpha = None
        self.r_ref = None

    def load_data(
            self,
            coord_fp: str = None,
            pressure_fp: str = None,
            pressure_col_idx: int = 1
    ):
        """
        Parameters
        ----------
        coord_fp : File path of coordinates
        pressure_fp : File path of CSV file.
        coord_format : For coordinate files in different formats. Currently only "ImageJ" is supported.
        normalize: Normalize all the points within a specific experiment. Each experiment
        is characterized by multiple images, with each having a set of points. Normalize ensures that
        all the points are consistent throughout the experiments
        pressure_col_idx: Row to read off the column pressure.

        Returns
        -------

        """
        x_coords, y_coords, pressures = load_keypoints_pressures(coord_fp,
                                                                 pressure_fp,
                                                                 self.normalize,
                                                                 pressure_col_idx,
                                                                 num_points_per_image=self.num_points_per_image)

        self.x_coords, self.y_coords, self.pressures = x_coords, y_coords, pressures
        return x_coords, y_coords, pressures

    def get_capillary_geometry(self, x_coords, y_coords):

        if self.normalize:
            x_coords = normalize_keypoint(x_coords)
            y_coords = normalize_keypoint(y_coords)

        top_line = get_line_param(
            x_coords[:, 1:3].flatten(), y_coords[:, 1:3].flatten()
        )
        # get line fit from x3, x4
        bot_line = get_line_param(
            x_coords[:, 3:5].flatten(), y_coords[:, 3:5].flatten()
        )

        # middle lines
        mid_intercept = np.divide(bot_line.intercept + top_line.intercept, 2)
        grad_intercept = np.divide(bot_line.grad + top_line.grad, 2)
        middle_line = Line(grad=grad_intercept, intercept=mid_intercept)

        intersect = get_intersection(top_line, bot_line)
        alpha = self.calc_alpha(top_line, bot_line)
        alpha_v2 = self.calc_alpha_v2(top_line, bot_line)

        return top_line, bot_line, middle_line, intersect, alpha_v2

    def process_particle_points(self, x_coords, y_coords) -> Tuple[List[CapillaryPoints], List[CapillaryPoints], float]:
        # hold our private vars
        top_line, bot_line, mid_line, intersect, alpha = self.get_capillary_geometry(x_coords, y_coords)
        capillaries = self.get_capillary_points(x_coords, y_coords)

        for idx, cap in enumerate(capillaries):
            projections = [None, top_line, top_line, bot_line, bot_line, mid_line, mid_line]
            capillaries[idx] = cap.project_points_to_lines(projections)

        # middle line projection
        mid_capillaries = deepcopy(capillaries)

        for idx, mid_cap in enumerate(mid_capillaries):
            mid_cap.p0 = Point(0, 0)
            projections = [mid_line] * len(mid_cap)
            mid_capillaries[idx] = mid_cap.project_points_to_lines(projections)

        assert len(capillaries) == len(mid_capillaries)

        return capillaries, mid_capillaries, alpha

    def get_l_band(self, tip_distance: dict):
        _d_band_front = (tip_distance["d1"] + tip_distance["d3"]) / 2
        _d_band_back = (tip_distance["d2"] + tip_distance["d4"]) / 2
        return _d_band_back - _d_band_front

    def get_r_band(self, cp: CapillaryPoints, mcp: CapillaryPoints):
        r_band_front = (dist_2D(cp.p1, mcp.p1) + dist_2D(cp.p3, mcp.p3)) / 2
        r_band_back = (dist_2D(cp.p2, mcp.p2) + dist_2D(cp.p4, mcp.p4)) / 2
        return r_band_front, r_band_back, (r_band_back + r_band_front) / 2

    def get_w_band(self, cp):
        """
        Get slant length from a CapillaryPoints object

        Parameters
        ----------
        cp : CapillaryPoints Object.

        Returns
        -------
        Slant length as a float.
        """
        return (dist_2D(cp.p1, cp.p2) + dist_2D(cp.p3, cp.p4)) / 2

    def get_volume(self, tip_distance: dict, cp: CapillaryPoints, mcp: CapillaryPoints):

        l_band = self.get_l_band(tip_distance)
        r_band_front, r_band_back, r_band = self.get_r_band(cp, mcp)

        h_cap_front = (tip_distance["d1"] + tip_distance["d3"]) / 2 - tip_distance["d5"]
        h_cap_back = (tip_distance["d6"] - (tip_distance["d2"] + tip_distance["d4"]) / 2)

        a_cap_front = (dist_2D(mcp.p5, cp.p1) + dist_2D(mcp.p5, cp.p3)) / 2
        a_cap_back = (dist_2D(mcp.p6, cp.p2) + dist_2D(mcp.p6, cp.p4)) / 2

        v_cap_front = np.pi * h_cap_front * (3 * a_cap_front ** 2 + h_cap_front ** 2) / 6
        v_cap_back = np.pi * h_cap_back * (3 * a_cap_back ** 2 + h_cap_back ** 2) / 6

        total_length = l_band / (1 - (r_band_front / r_band_back))
        # volume of frustum
        v_band = np.pi / 3 * (r_band_back ** 2 * total_length - r_band_front ** 2 * (total_length - l_band))
        return v_band + v_cap_back + v_cap_front

    def get_band_area(self, r_b, r_f, w_band):
        return np.pi * (r_f + r_b) * w_band

    def calculate(self, x_coords, y_coords, pressures):

        capillaries, mid_capillaries, alpha = self.process_particle_points(x_coords, y_coords)
        self.alpha = alpha
        # get distances for each image and append to list.
        num_images = len(capillaries)
        self.l_bands = np.zeros(num_images)
        self.r_bands = np.zeros(num_images)
        self.lengths = np.zeros(num_images)
        self.volumes = np.zeros(num_images)
        self.areas = np.zeros(num_images)

        if self.pressures is None:
            self.pressures = pressures

        self.wall_pressures = np.zeros(num_images)
        self.wall_min_pressures = np.zeros(num_images)
        self.avg_pressures = np.zeros(num_images)

        for idx in range(len(capillaries)):
            cp, mcp = capillaries[idx], mid_capillaries[idx]
            tip_distance = mcp.calc_distances()

            l_band = self.get_l_band(tip_distance)
            w_band = self.get_w_band(cp)  # avg slant length
            r_band_front, r_band_back, r_band = self.get_r_band(cp, mcp)
            a_band = self.get_band_area(r_band_back, r_band_front, w_band)

            length = dist_2D(mcp.p5, mcp.p6)

            p_wall, p_avg, p_wall_min_p = self.calc_pressures(
                pressures[idx], alpha, r_band_back, a_band
            )
            self.lengths[idx] = length
            self.r_bands[idx] = r_band
            self.volumes[idx] = self.get_volume(tip_distance, cp, mcp)
            self.areas[idx] = np.pi * (r_band ** 2)
            self.l_bands[idx] = l_band
            self.wall_pressures[idx] = p_wall
            self.wall_min_pressures[idx] = p_wall_min_p
            self.avg_pressures[idx] = p_avg

        self._calc_strains(num_images)

        return self.eps_g, self.v_strain, self.wall_min_pressures, self.avg_pressures

    def get_bands(self, x_coords, y_coords, normalize=True) -> np.ndarray:
        """
        Get bands from keypoints
        Parameters
        ----------
        normalize
        x_coords : list or array of x-coordinate of keypoints
        y_coords : list or array of y-coordinate of keypoints

        Returns
        -------

        """
        if normalize:
            x_coords = normalize_keypoint(x_coords)
            y_coords = normalize_keypoint(y_coords)

        cp, mcp, alpha = self.process_particle_points(x_coords, y_coords)
        # assert len(x_coords) % 7 == 0 and len(y_coords) % 7 == 0
        bands = np.zeros((len(cp), 2))
        for idx, (i_cp, i_mcp) in enumerate(zip(cp, mcp)):
            tip_d = i_mcp.calc_distances()
            _, _, r_band = self.get_r_band(i_cp, i_mcp)
            l_band = self.get_l_band(tip_d)
            bands[idx][0] = r_band
            bands[idx][1] = l_band
        return bands

    @staticmethod
    def calc_pressures(pressure, alpha, r_band_back, a_band):
        """Calculate the wall, average and pressure difference

        Parameters
        ----------
        pressure : Applied pressure
            _description_
        alpha : Angle in rad
            _description_
        r_band_back : Radius of back band
            _description_
        a_band : Area of band
            _description_

        Returns
        -------
        Tuple
            wall, average and pressure difference values
        """
        f_p = pressure * (
                np.pi * r_band_back ** 2
        )  # force onto back portion of particle by pressure
        f_wall = f_p / np.sin(alpha)
        p_wall = f_wall / a_band
        p_wall_min_p = p_wall - pressure
        p_avg = (2 * p_wall + pressure) / 3  # x, y and z pressures.

        return p_wall, p_avg, p_wall_min_p

    def _calc_strains(self, num_images):
        # Area strain : due to relative change in cross sectional area,

        eps_1 = np.zeros(num_images)
        v_strain_1 = np.zeros(num_images)  # volumetric strain due to pure compression
        # eps_z = np.zeros(self.num_images)  # strain in the z direction
        # set reference at index 0

        r_ref = (3 * self.volumes[0] / 4 / np.pi) ** (1 / 3)
        a_ref = np.pi * r_ref ** 2

        self.eps_r = (self.r_bands[0] - self.r_bands) \
                     / self.r_bands[0]  # strain in the r direction

        # a_strain = divide_by_zero(a_ref - self.areas, self.areas)
        self.eps_z = (self.l_bands[0] - self.l_bands) / self.l_bands[0]
        length_strain = (self.lengths[0] - self.lengths) / self.lengths[0]

        self.v_strain = (self.volumes[0] - self.volumes) / self.volumes[0]
        eps_1[1:] = 1 - np.exp(-(self.pressures[1:] - self.pressures[0]) / (6 * BULK_MODULUS))
        v_strain_1[1:] = 1 - np.exp(-(self.pressures[1:] - self.pressures[0]) / (2 * BULK_MODULUS))
        v_strain_2 = divide_by_zero(self.v_strain - v_strain_1,
                                    1 - v_strain_1)  # volumetric strain after subtracting strain from pure compression
        length_strain_2 = divide_by_zero(length_strain - eps_1, 1 - eps_1)
        eps_r_2 = divide_by_zero(self.eps_r - eps_1, 1 - eps_1)

        self.eps_g = 2 * (self.eps_r - self.eps_z)
        self.K_compressive = calc_compressive_modulus(
            self.wall_pressures, self.pressures, self.eps_r, self.eps_z
        )  # compressive modulus
        self.G_shear = calc_shear_modulus(
            self.wall_pressures, self.pressures, self.eps_r, self.eps_z
        )

        # step wise evaluation of compressive and shear stress
        delta_p_wall = self.wall_pressures - self.wall_pressures[0]
        delta_p = self.pressures - self.pressures[0]

    @staticmethod
    def get_capillary_points(x_coords, y_coords) -> List[CapillaryPoints]:
        capillary_points = []
        for c_x, c_y in zip(x_coords, y_coords):
            points = [Point(_x, _y) for _x, _y in zip(c_x, c_y)]
            capillary_points.append(CapillaryPoints(*points))

        return capillary_points

    @staticmethod
    def calc_alpha(top_line: Line, bot_line: Line):
        return np.abs(top_line.grad - bot_line.grad) / 2

    @staticmethod
    def calc_alpha_v2(top_line: Line, bot_line: Line):
        num = top_line.grad - bot_line.grad
        den = 1 + top_line.grad * bot_line.grad
        return np.abs(np.arctan(num / den)) / 2

    def plot_g_k(self, x_coords, y_coords, pressures):

        self.calculate(x_coords, y_coords, pressures)

        # shear modulus
        G_line = get_line_param(self.eps_g, self.wall_min_pressures)

        # compressive modulus
        K_line = get_line_param(self.v_strain, self.avg_pressures)

        fig, [ax_G, ax_K] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax_G.scatter(self.eps_g, self.wall_min_pressures)
        ax_G.set_xlabel(r"2($\epsilon_r - \epsilon_z$)")
        # ax_G.ticklabel_format(axis='y', style='sci', scilimits=(3,1))
        ax_G.set_ylabel("Minimum wall pressure")
        ax_G.plot(
            self.eps_g,
            G_line(self.eps_g),
            label="Best Fit Line",
            linestyle="--",
        )
        ax_G.set_title(
            "Fit of Shear Modulus,\n" + r"G $\approx$" + f"{int(G_line.grad)} Pa"
        )
        ax_G.legend()
        # ax_G.set_aspect(1)

        ax_K.scatter(self.v_strain, self.avg_pressures)
        ax_K.set_xlabel(r"$\epsilon_V$ ")
        ax_K.set_ylabel("Average Pressure")
        ax_K.plot(
            self.v_strain,
            K_line(self.v_strain),
            label="Best Fit Line",
            linestyle="--",
        )
        ax_K.set_title(
            "Fit of Compressive Modulus,\n" + r"K $\approx$" + f"{int(K_line.grad)} Pa"
        )
        ax_K.legend()
        plt.tight_layout()
        plt.show()


def calc_compressive_modulus(p_wall, p, eps_r, eps_z):
    """
    Calculate local K (compressive modulus) according to formula

    :param p_wall: Pressure on the capillary wall
    :param p: Applied pressure
    :param eps_r: strain in radial direction
    :param eps_z: strain in longitudinal direction
    :return: Compressive modulus
    """
    return divide_by_zero(1 / 3 * (2 * p_wall + p), 2 * eps_r + eps_z)


def calc_shear_modulus(p_wall, p, eps_r, eps_z):
    """
    Calculate local G (shear modulus) according to formula

    :param p_wall: Pressure on the capillary wall
    :param p: Applied pressure
    :param eps_r: strain in radial direction
    :param eps_z: strain in longitudinal direction
    :return: Shear modulus
    """
    return divide_by_zero(0.5 * (p_wall - p), eps_r - eps_z)


def calc_elastic_modulus(G: float, K: float):
    """
    Calculate local E (elastic modulus) according to formula

    :param G:
    :param K:
    :return:
    """
    return np.divide(9 * K * G, G + 3 * K)


def calc_p_wall(r_band, l_band, p, alpha, is_rad=False):
    if not is_rad:
        alpha = np.deg2rad(alpha)

    return np.divide(2, np.sin(alpha)) * np.divide(r_band, l_band) * p


def calc_V(r1, r2, b1, b2, l_band):
    # split volume into 3 components
    # 2 half-ellipsoid + 1 conical frustum
    vol_e1 = np.divide(2, 3) * np.pi * r1 ** 2 * b1
    vol_e2 = np.divide(2, 3) * np.pi * r2 ** 2 * b2
    vol_cf = np.divide(np.pi * l_band, 3) * (r1 ** 2 + r1 * r2 + r2 ** 2)
    return vol_e1 + vol_e2 + vol_cf

def get_eps_rz_from_eps_gk(eps_gk:np.ndarray):
    A_inv = 1/3 * np.array([[1 , 1],[-2, 1]])
    eps = A_inv.dot(eps_gk)  # [er, ez]

    return eps

if __name__ == "__main__":
    sb = CapillaryStressBalance()
    x, y, p = sb.load_data(
        coord_fp="dataset/Inc_press_2/Inc_press_2.txt",
        pressure_fp="dataset/Inc_press_2/Pressure_mm_H20_dummy.txt",
    )
    sb.plot_g_k(x, y, p)
    l, r = sb.get_bands(x, y)

    decompose_l_r_band(sb.alpha, l, r)
    # sb.l_bands, sb.r_bands
