import numpy as np
from dataclasses import dataclass
from typing import List, Union
from copy import deepcopy
from src.utils.constants import PRESSURE_CONSTANT_MULTIPLIER, BULK_MODULUS
import matplotlib.pyplot as plt

from src.utils.transforms import divide_by_zero


@dataclass
class Point:
    x: Union[int, None]
    y: Union[int, None]


@dataclass
class CapillaryPoints:
    p0: Point
    p1: Point
    p2: Point
    p3: Point
    p4: Point
    p5: Point
    p6: Point

    def calc_distances(self) -> dict:
        # for
        # d1 = self.dist_2D(cap_points.p1, cap_points.0)
        distances = {}
        distances['d1'] = dist_2D(self.p1, self.p0)
        distances['d2'] = dist_2D(self.p2, self.p0)
        distances['d3'] = dist_2D(self.p3, self.p0)
        distances['d4'] = dist_2D(self.p4, self.p0)
        distances['d5'] = dist_2D(self.p5, self.p0)
        distances['d6'] = dist_2D(self.p6, self.p0)

        assert len(distances) == 6
        return distances


def dist_2D(p1: Point, p2: Point):
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


class CapillaryStressBalance:

    def __init__(self, num_points_per_image: int = 7):
        self.x_coords = None
        self.y_coords = None
        self.pressures = None
        self.capillary_points: Union[None, List[CapillaryPoints]] = None
        self.num_images = None

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

    def load_data(self, coord_fp: str,
                  pressure_fp: str,
                  coord_format="imagej",
                  normalize: bool = True,
                  pressure_col_idx: int = 1):

        # default d type
        coord_dtype = int

        if coord_format == "imagej":
            with open(coord_fp) as f:
                header = f.readline().split()
                coord_dtype = {'names': tuple(header),
                               'formats': (('i4',) * len(header))}
        else:
            raise (Exception(f'Format {coord_format} not supported'))

        coord_data = np.loadtxt(coord_fp, skiprows=1, dtype=coord_dtype)
        self.num_images = len(coord_data) // self.num_points_per_image

        self.pressures = np.loadtxt(pressure_fp, skiprows=1, dtype='f4',
                                    usecols=pressure_col_idx) * PRESSURE_CONSTANT_MULTIPLIER

        if self.num_images != len(self.pressures):
            print('Number of image slices not equal to supplied pressures. Data will be truncated accordingly.')
            min_slices = min(self.num_images, len(self.pressures))
            coord_data = coord_data[:(min_slices * self.num_points_per_image)]
            self.pressures = self.pressures[:min_slices]

        self.x_coords = coord_data['x'].reshape(-1, self.num_points_per_image)
        self.y_coords = coord_data['y'].reshape(-1, self.num_points_per_image)

        if normalize:
            self.x_coords = self._normalize(self.x_coords)
            self.y_coords = self._normalize(self.y_coords)

        # self.capillary_points = [CapillaryPoints()]

    def _normalize(self, coords, ref_idx=0):
        return coords - coords[:, ref_idx][:, np.newaxis]

    def calculate(self):
        # hold our private vars

        # calibration of top and bottom lines
        # get line fit from x1, x2, polyfit returns _increasing_ degree (different from matlab)
        a_top, b_top = self._get_line_fits(self.x_coords[:, 1:3].flatten(),
                                           self.y_coords[:, 1:3].flatten())
        # get line fit from x3, x4
        a_bot, b_bot = self._get_line_fits(self.x_coords[:, 3:5].flatten(),
                                           self.y_coords[:, 3:5].flatten())

        # get contact angle
        self.alpha = self._calc_alpha(b_top, b_bot)

        # middle lines
        b_mid = np.divide(b_bot + b_top, 2)
        a_mid = np.divide(a_bot + a_top, 2)

        # intersection between top and bottom (from matlab)
        p_intersect = Point(None, None)
        p_intersect.x = np.divide(a_bot - a_top, b_top - b_bot)
        p_intersect.y = a_top + p_intersect.x * b_top

        capillary_points = self._get_capillary_points(self.x_coords, self.y_coords)

        for cap in capillary_points:
            cap.p1 = self.projection(a_top, b_top, cap.p1)
            cap.p2 = self.projection(a_top, b_top, cap.p2)
            cap.p3 = self.projection(a_bot, b_bot, cap.p3)
            cap.p4 = self.projection(a_bot, b_bot, cap.p4)
            cap.p5 = self.projection(a_mid, b_mid, cap.p5)
            cap.p6 = self.projection(a_mid, b_mid, cap.p6)

        # middle line projection

        mid_point_capillary_points = deepcopy(capillary_points)

        for mid_caps in mid_point_capillary_points:
            mid_caps.p0 = self.projection(a_mid, b_mid, Point(0, 0))
            mid_caps.p1 = self.projection(a_mid, b_mid, mid_caps.p1)
            mid_caps.p2 = self.projection(a_mid, b_mid, mid_caps.p2)
            mid_caps.p3 = self.projection(a_mid, b_mid, mid_caps.p3)
            mid_caps.p4 = self.projection(a_mid, b_mid, mid_caps.p4)
            mid_caps.p5 = self.projection(a_mid, b_mid, mid_caps.p5)
            mid_caps.p6 = self.projection(a_mid, b_mid, mid_caps.p6)

        assert len(capillary_points) == len(mid_point_capillary_points)

        tip_distances = self._calc_tip_distances(mid_point_capillary_points)
        assert len(tip_distances) == len(mid_point_capillary_points)
        # get distances for each image and append to list.

        self.l_bands = np.zeros(self.num_images)
        self.r_bands = np.zeros(self.num_images)
        self.lengths = np.zeros(self.num_images)
        self.volumes = np.zeros(self.num_images)
        self.areas = np.zeros(self.num_images)

        self.wall_pressures = np.zeros(self.num_images)
        self.wall_min_pressures = np.zeros(self.num_images)
        self.avg_pressures = np.zeros(self.num_images)

        for idx in range(len(capillary_points)):
            tip_distance = tip_distances[idx]
            _d_band_front = (tip_distance['d1'] + tip_distance['d3']) / 2
            _d_band_back = (tip_distance['d2'] + tip_distance['d4']) / 2
            l_band = _d_band_back - _d_band_front
            self.l_bands[idx] = l_band

            cp, mcp = capillary_points[idx], mid_point_capillary_points[idx]
            w_band = (dist_2D(cp.p1, cp.p2) + dist_2D(cp.p3, cp.p4)) / 2 # avg slant length

            r_band_front = (dist_2D(cp.p1, mcp.p1) + dist_2D(cp.p3, mcp.p3)) / 2
            r_band_back = (dist_2D(cp.p2, mcp.p2) + dist_2D(cp.p4, mcp.p4)) / 2
            r_band = (r_band_back + r_band_front) / 2
            self.r_bands[idx] = r_band

            # c_temp = (r_band_back - r_band_front) / w_band

            # why not use conical frustum?
            a_band = np.pi * (r_band_front + r_band_back) * w_band
            # a_band = 2 * np.pi * (r_band_front * w_band + c_temp / 2 * w_band ** 2)
            length = dist_2D(mcp.p5, mcp.p6)
            self.lengths[idx] = length

            # calculate volumes
            h_cap_front = (tip_distance['d1'] + tip_distance['d3']) / 2 - tip_distance['d5']
            h_cap_back = tip_distance['d6'] - (tip_distance['d2'] + tip_distance['d4']) / 2

            a_cap_front = (dist_2D(mcp.p5, cp.p1) + dist_2D(mcp.p5, cp.p3)) / 2
            a_cap_back = (dist_2D(mcp.p6, cp.p2) + dist_2D(mcp.p6, cp.p4)) / 2

            v_cap_front = np.pi * h_cap_front * (3 * a_cap_front ** 2 + h_cap_front ** 2) / 6
            v_cap_back = np.pi * h_cap_back * (3 * a_cap_back ** 2 + h_cap_back ** 2) / 6

            total_length = l_band / (1 - (r_band_front / r_band_back))
            # volume of frustum
            v_band = np.pi / 3 * (r_band_back ** 2 * total_length - r_band_front ** 2 * (total_length - l_band))
            volume = v_band + v_cap_back + v_cap_front
            self.volumes[idx] = volume

            area = np.pi * (r_band ** 2)
            self.areas[idx] = area

            p_wall, p_avg, p_wall_min_p = self.calc_pressures(self.pressures[idx], self.alpha, r_band_back, a_band)

            self.wall_pressures[idx] = p_wall
            self.wall_min_pressures[idx] = p_wall_min_p
            self.avg_pressures[idx] = p_avg

        self.calc_strains()

    @staticmethod
    def calc_pressures(pressure, alpha, r_band_back, a_band):
        f_p = pressure * (np.pi * r_band_back ** 2)  # force onto back portion of particle by pressure
        f_wall = f_p / np.sin(alpha)
        p_wall = f_wall / a_band
        p_wall_min_p = p_wall - pressure
        # a_ratio = np.pi * r_band_back**2 / a_band
        p_avg = (2 * p_wall + pressure) / 3 # x, y and z pressures.

        return p_wall, p_avg, p_wall_min_p

    def calc_strains(self):
        # Area strain : due to relative change in cross sectional area,

        eps_1 = np.zeros(self.num_images)
        v_strain_1 = np.zeros(self.num_images)  # volumetric strain due to pure compression
        eps_z = np.zeros(self.num_images)  # strain in the z direction
        # set reference at index 0

        r_ref = (3 * self.volumes[0] / 4 / np.pi) ** (1 / 3)
        a_ref = np.pi * r_ref ** 2

        self.eps_r = (self.r_bands[0] - self.r_bands) / self.r_bands[0]  # strain in the r direction

        a_strain = divide_by_zero(a_ref - self.areas, self.areas)
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
        self.K_compressive = calc_K(self.wall_pressures, self.pressures, self.eps_r, self.eps_z)  # compressive modulus
        self.G_shear = calc_G(self.wall_pressures, self.pressures, self.eps_r, self.eps_z)

        # step wise evaluation of compressive and shear stress
        delta_p_wall = self.wall_pressures - self.wall_pressures[0]
        delta_p = self.pressures - self.pressures[0]
        K_compressive_stepwise = calc_K(delta_p_wall, delta_p, self.eps_r, self.eps_z)
        G_shear_stepwise = calc_G(delta_p_wall, delta_p, self.eps_r, self.eps_z)

        r_shape = self.r_bands[:-1] * (self.volumes[0] / self.volumes.T[:-1]) ** (1 / 3)
        # poisson_ratio = length_strain_2 / 4 / eps_r_2
        # G_div_K = 3 * (1- poisson_ratio) / (2 + 2 * poisson_ratio)

    def _calc_tip_distances(self, capillary_points: List[CapillaryPoints]):
        return [mid_caps.calc_distances() for mid_caps in capillary_points]

    @staticmethod
    def projection(a, b, p: Point) -> Point:
        c_temp = (1 + b ** 2) ** 0.5
        x_perp = -b / c_temp
        y_perp = 1 / c_temp
        x = np.divide(p.y - a - (y_perp / x_perp) * p.x, b - y_perp / x_perp)
        y = a + b * x

        return Point(x, y)

    @staticmethod
    def _get_capillary_points(x_coords, y_coords) -> List[CapillaryPoints]:
        capillary_points = []
        for x, y in zip(x_coords, y_coords):
            p0 = Point(x[0], y[0])
            p1 = Point(x[1], y[1])
            p2 = Point(x[2], y[2])
            p3 = Point(x[3], y[3])
            p4 = Point(x[4], y[4])
            p5 = Point(x[5], y[5])
            p6 = Point(x[6], y[6])
            capillary_points.append(CapillaryPoints(p0, p1, p2, p3, p4, p5, p6))

        return capillary_points

    @staticmethod
    def _calc_alpha(b_top_from_fit, b_bott_from_fit):
        return np.abs(b_top_from_fit - b_bott_from_fit) / 2

    @staticmethod
    def _get_line_fits(x_coords, y_coords) -> list:
        # note, coeffients are presented in increasing degree
        # so returns (slope, intercept)
        return np.polynomial.Polynomial.fit(x_coords, y_coords, 1, domain=[0, 1], window=[0, 1]).coef.tolist()

    def plot_figures(self):

        assert self.wall_min_pressures is not None
        assert self.v_strain is not None
        assert self.avg_pressures is not None

        # shear modulus
        intercept_G, slope_G = self._get_line_fits(self.eps_g, self.wall_min_pressures)

        # compressive modulus
        intercept_K, slope_K = self._get_line_fits(self.v_strain, self.avg_pressures)

        fig, [ax_G, ax_K] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax_G.scatter(self.eps_g, self.wall_min_pressures)
        ax_G.set_xlabel(r'2($\epsilon_r - \epsilon_z$)')
        # ax_G.ticklabel_format(axis='y', style='sci', scilimits=(3,1))
        ax_G.set_ylabel('Minimum wall pressure')
        ax_G.plot(self.eps_g, intercept_G + slope_G * self.eps_g, label='Best Fit Line', linestyle='--')
        ax_G.set_title('Fit of Shear Modulus,\n' + r'G $\approx$' + f'{int(slope_G)} Pa')
        ax_G.legend()
        # ax_G.set_aspect(1)

        ax_K.scatter(self.v_strain, self.avg_pressures)
        ax_K.set_xlabel(r'$\epsilon_V$ ')
        ax_K.set_ylabel('Average Pressure')
        ax_K.plot(self.v_strain, intercept_K + slope_K * self.v_strain, label='Best Fit Line', linestyle='--')
        ax_K.set_title('Fit of Compressive Modulus,\n' + r'K $\approx$' + f'{int(slope_K)} Pa')
        ax_K.legend()
        # ax_K.set_aspect(1)

        plt.tight_layout()
        plt.show()


def calc_K(p_wall, p, eps_r, eps_z):
    """
    Calculate K (compressive modulus) according to formula

    :param p_wall: Pressure on the capillary wall
    :param p: Applied pressure
    :param eps_r: strain in radial direction
    :param eps_z: strain in longitudinal direction
    :return: Compressive modulus
    """
    return divide_by_zero(1 / 3 * (2 * p_wall + p), 2 * eps_r + eps_z)


def calc_G(p_wall, p, eps_r, eps_z):
    """
    Calculate G (shear modulus) according to formula

    :param p_wall: Pressure on the capillary wall
    :param p: Applied pressure
    :param eps_r: strain in radial direction
    :param eps_z: strain in longitudinal direction
    :return: Shear modulus
    """
    return divide_by_zero(0.5 * (p_wall - p), eps_r - eps_z)


def calc_E(G: float, K: float):
    """
    Calculate E (elastic modulus) according to formula

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


if __name__ == "__main__":
    sb = CapillaryStressBalance()
    sb.load_data(coord_fp="dataset/sample/sample_annotated_data.txt", pressure_fp="dataset/sample/sample_pressures.txt")
    sb.calculate()
    sb.plot_figures()
