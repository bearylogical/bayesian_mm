import numpy as np



def calc_K(p_wall, p, eps_r, eps_z):
    """
    Calculate K (compressive modulus) according to formula

    :param p_wall:
    :param p:
    :param eps_r:
    :param eps_z:
    :return:
    """
    return np.divide(1 / 3 * (2 * p_wall + p), 2 * eps_r + eps_z)


def calc_G(p_wall, p, eps_r, eps_z):
    """
    Calculate G (shear modulus) according to formula

    :param p_wall:
    :param p:
    :param eps_r:
    :param eps_z:
    :return:
    """
    return np.divide(0.5 * (p_wall - p), eps_r - eps_z)


def calc_E(G:float, K:float):
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
    vol_cf = np.divide(np.pi * L_band, 3) * (r1 ** 2 + r1 * r2 + r2 ** 2)
    return vol_e1 + vol_e2 + vol_cf
