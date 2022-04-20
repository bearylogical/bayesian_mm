from copy import deepcopy
from typing import Union

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.inference.models.model import IsotropicModel
from src.utils.shapes.capillary import CapillaryImage, get_outline
from src.utils.geometry import get_line_param


def get_eps(inputs, rl_obs):
    r_obs, l_obs = rl_obs[:, 0], rl_obs[:, 1]
    r_0, l_0, p_0 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    eps_r = (r_0 - r_obs) / r_0
    eps_z = (l_0 - l_obs) / l_0
    return eps_z, eps_r


def plot_r_l_scatter(y_experiments, m_observations):
    plt.xlabel(r'$R_{\mathrm{band}}$')
    plt.ylabel(r'$L_{\mathrm{band}}$')
    # for idx, (x_i, y_i) in enumerate(zip(y_experiments[:,0], y_experiments[:,1])):
    #     plt.text(x_i, y_i+0.001, idx)
    for exp in range(m_observations):
        plt.scatter(y_experiments[exp::m_observations, 0],
                    y_experiments[exp::m_observations, 1], color=plt.cm.tab10(exp),
                    label=f'Experiment {exp + 1}')
    plt.legend()
    plt.show()


def plot_g_k_bands(x, y_experiments, p_factor, m_observations):
    eps_z, eps_r = get_eps(x, y_experiments)
    p_wall = x[:, 2] * p_factor
    pressure = x[:, 2]
    del_p = (p_wall - pressure)
    p_avg =  1 / 3 * (2 * p_wall + pressure)
    fig, [ax_G, ax_K] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    mean_g = get_line_param(2 * (eps_r - eps_z), del_p)
    mean_k = get_line_param(2 * eps_r + eps_z, p_avg)
    ax_G.plot(
            2 * (eps_r - eps_z),
            mean_g(2 * (eps_r - eps_z)),
            label=f"Linear fit",
            linestyle="--",
            color='black'
        )
    ax_K.plot(
            2 * eps_r + eps_z,
            mean_k(2 * eps_r + eps_z),
            label=f"Linear fit",
            linestyle="--",
            color='black'
        )
    ax_G.set_xlabel(r"2($\epsilon_r - \epsilon_z$)")
    # ax_G.ticklabel_format(axis='y', style='sci', scilimits=(3,1))
    ax_G.set_ylabel("Minimum wall pressure")
    ax_G.set_title(
        "Fit of Shear Modulus,\n" + r"G $\approx$" + f"{mean_g.grad:.3f} x10^5 Pa"
    )

    ax_K.set_title(
        "Fit of Compressive Modulus,\n" + r"K $\approx$" + f"{mean_k.grad:.3f} x10^5 Pa"
    )

    ax_K.set_xlabel(r"$\epsilon_V$ ")
    ax_K.set_ylabel("Average Pressure")

    for exp in range(m_observations):
        _eps_r = eps_r[exp::m_observations]
        _eps_z = eps_z[exp::m_observations]
        _del_p = del_p[exp::m_observations]
        _p_avg = p_avg[exp::m_observations]
        g_line = get_line_param(2 * (_eps_r - _eps_z), _del_p)
        k_line = get_line_param(2 * _eps_r + _eps_z, _p_avg)

        ax_G.scatter(2 * (_eps_r - _eps_z), _del_p, color=plt.cm.tab10(exp))

        ax_G.plot(
            2 * (_eps_r - _eps_z),
            g_line(2 * (_eps_r - _eps_z)),
            label=f"Experiment {exp + 1}",
            linestyle="--",
            color=plt.cm.tab10(exp)
        )

        # ax_G.set_aspect(1)

        ax_K.scatter(2 * _eps_r + _eps_z, _p_avg, color=plt.cm.tab10(exp))

        ax_K.plot(
            2 * _eps_r + _eps_z,
            k_line(2 * _eps_r + _eps_z),
            label=f"Experiment {exp + 1}",
            linestyle="--",
            color=plt.cm.tab10(exp)
        )
    ax_K.legend()
    ax_G.legend()
    plt.tight_layout()
    plt.show()


def plot_capillary_observations(initial_pos: Union[np.ndarray, list],
                                observed_pos: Union[np.ndarray, list],
                                alpha: float,
                                noise_obs=None, figsize=(20, 6)):
    initial = CapillaryImage(
        theta=alpha, l_band=initial_pos[1], r_band=initial_pos[0], img_size=(600, 400), taper_cutoff=200, is_deg=False,
        fill_alpha_outer=0
    )
    initial_img = Image.new(mode="L", size=initial.dim, color=255)
    initial.generate_image(initial_img, is_annotate=False)
    imgs = [initial_img]
    for pos in observed_pos:
        observed_cap = CapillaryImage(
            theta=alpha, l_band=pos[1], r_band=pos[0], img_size=(600, 400), taper_cutoff=200, is_deg=False)
        obs_image = Image.new(mode="L", size=observed_cap.dim, color=255)
        observed_cap.generate_image(obs_image, is_annotate=False)
        t_img = deepcopy(obs_image)
        imgs.append(t_img)

    nrow = len(imgs) // 3 if len(imgs) % 3 == 0 else len(imgs) // 3 + 1
    fig, axes = plt.subplots(ncols=3, nrows=nrow, figsize=figsize)
    for idx, img in enumerate(imgs):
        if idx == 0:
            seq_text = "Initial"
        else:
            seq_text = f"Sequence No. : {idx}"
        axes.flat[idx].text(10, 380, seq_text)
        axes.flat[idx].imshow(img, cmap='gray', aspect='auto')
        if noise_obs is not None and idx < len(noise_obs):
            noise_ob = noise_obs[idx + 1]
            noisy_obs_outline = get_outline(noise_obs[1], noise_ob[0], initial)
            axes.flat[idx + 1].plot(noisy_obs_outline[0],
                                    noisy_obs_outline[1],
                                    alpha=0.5, color='red',
                                    label='Noise')
            axes.flat[idx + 1].legend()
    plt.show()
