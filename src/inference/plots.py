import random
from copy import deepcopy
from pathlib import Path
from typing import Union
import logging
import corner
import emcee
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from src.inference.models.model import IsotropicModel, BayesModel
from src.utils.shapes.capillary import CapillaryImage, get_outline
from src.utils.geometry import get_line_param

logger = logging.getLogger('bayesian_nn')

default_capillary_kwargs = dict(img_size=(600, 400),
                                taper_cutoff=200, is_deg=False,
                                fill_alpha_inner=0.8, fill_alpha_outer=0)

rng = np.random.default_rng(42)


def get_samples(sampler: emcee.EnsembleSampler,
                model: BayesModel) -> np.ndarray:
    acts = sampler.get_autocorr_time()
    for param_name, act in zip(model.param_names, acts):
        logger.info('auto-correlation ({}): {}'.format(param_name, act))

    # burn-in and thinning
    act_max = int(np.max(acts))
    flat_samples = sampler.get_chain(discard=2 * act_max, thin=act_max // 2, flat=True)

    return flat_samples


def get_eps(inputs, rl_obs):
    r_obs, l_obs = rl_obs[:, 0], rl_obs[:, 1]
    r_0, l_0, p_0 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    eps_r = (r_0 - r_obs) / r_0
    eps_z = (l_0 - l_obs) / l_0
    return eps_z, eps_r


def plot_r_l_scatter(y_experiments,
                     m_observations,
                     save_path: Path = None, ):
    plt.xlabel(r'$R_{\mathrm{band}}$')
    plt.ylabel(r'$L_{\mathrm{band}}$')
    # for idx, (x_i, y_i) in enumerate(zip(y_experiments[:,0], y_experiments[:,1])):
    #     plt.text(x_i, y_i+0.001, idx)
    for exp in range(m_observations):
        plt.scatter(y_experiments[exp::m_observations, 0],
                    y_experiments[exp::m_observations, 1], color=plt.cm.tab10(exp),
                    label=f'Experiment {exp + 1}')
    plt.legend()

    if save_path:
        plt.savefig(save_path / f'r_l_scatter.png')

    plt.show()


def plot_g_k_bands(x, y_experiments, p_factor, m_observations, save_path: Path = None, ):
    eps_z, eps_r = get_eps(x, y_experiments)
    p_wall = x[:, 2] * p_factor
    pressure = x[:, 2]
    del_p = (p_wall - pressure)
    p_avg = 1 / 3 * (2 * p_wall + pressure)
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

    if save_path:
        plt.savefig(save_path / f'g_k_bands.png')

    plt.show()


def plot_capillary_observations(initial_pos: Union[np.ndarray, list],
                                observed_pos: Union[np.ndarray, list],
                                alpha: float,
                                noise_obs=None,
                                save_path: Path = None,
                                figsize=(20, 6)):
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
        if noise_obs is not None and idx > 0:
            noise_ob = noise_obs[idx - 1]
            noisy_obs_outline = get_outline(noise_ob[1], noise_ob[0], initial)
            axes.flat[idx].plot(noisy_obs_outline[0],
                                noisy_obs_outline[1],
                                alpha=0.5, color='red',
                                label='Noise')
            axes.flat[idx].legend()

    if save_path:
        plt.savefig(save_path / f'capillary_obs.png')

    plt.show()


def plot_samples(inputs: np.ndarray,
                 obs: np.ndarray,
                 flat_chain: np.ndarray,
                 alpha: float,
                 model: BayesModel,
                 ax: plt.Axes = None,
                 num_samples=500):
    cap_object = CapillaryImage(
        theta=alpha, l_band=obs[1], r_band=obs[0], img_size=(600, 400),
        taper_cutoff=200, is_deg=False, fill_alpha_inner=0.8, fill_alpha_outer=0)
    predicted_l_r_bands = np.array(
        [model.predict(sample.tolist(), np.expand_dims(inputs, axis=0)) for sample in flat_chain])
    temp_image = Image.new(mode="L", size=cap_object.dim, color=255)
    cap_object.generate_image(temp_image, is_annotate=False)
    num_rows_samples = len(predicted_l_r_bands)
    rand_idxs = rng.choice(num_rows_samples, size=num_samples, replace=False)
    for idx, sample in enumerate(predicted_l_r_bands[rand_idxs, :]):
        try:
            _x_coords, _y_coords = get_outline(sample.flatten()[1], sample.flatten()[0], cap_object)
            if idx == 0:
                ax.plot(_x_coords, _y_coords, alpha=0.01, color='blue', label='Predicted Shape from Samples')
            else:
                ax.plot(_x_coords, _y_coords, alpha=0.01, color='blue')
        except ValueError:
            pass
    mle_soln = model.max_likelihood(np.expand_dims(inputs, axis=0), obs)
    mle_rlbands = model.predict(mle_soln, np.expand_dims(inputs, axis=0))
    _x_coords_mle, _y_coords_mle = get_outline(mle_rlbands.flatten()[1], mle_rlbands.flatten()[0], cap_object)
    ax.plot(_x_coords_mle, _y_coords_mle, alpha=0.5, color='red', label='MLE', ls='--')
    ax.legend()
    ax.imshow(temp_image, cmap="gray", aspect="auto")


def plot_chain_obs(inputs: np.ndarray,
                   obs: np.ndarray,
                   flat_chain: np.ndarray,
                   alpha: float,
                   model: BayesModel,
                   num_show: int = 5,
                   save_path: Path = None,
                   figsize=(20, 12)):
    nrow = num_show // 3 if num_show % 3 == 0 else num_show // 3 + 1
    num_rows_samples = len(inputs)
    rand_idxs = rng.choice(num_rows_samples, size=num_show, replace=False)
    fig, axes = plt.subplots(ncols=3, nrows=nrow, figsize=figsize)
    for idx, (r_input, r_obs) in enumerate(zip(inputs[rand_idxs], obs[rand_idxs])):
        plot_samples(r_input, r_obs, flat_chain, alpha=alpha, ax=axes.flat[idx], model=model)

    if save_path:
        plt.savefig(save_path / 'sample_overlay.png')

    plt.show()


def plot_mean_shape(inputs,
                    obs: np.ndarray,
                    flat_chain: np.ndarray,
                    model: IsotropicModel,
                    alpha: float,
                    r_idx: int = 1,
                    save_path: Path = None,
                    figsize=(10, 6)):
    plt.figure(figsize=figsize)

    observed_cap = CapillaryImage(
        theta=alpha, l_band=obs[r_idx, 1], r_band=obs[r_idx, 0], img_size=(600, 400),
        taper_cutoff=200, is_deg=False, fill_alpha_inner=0.8, fill_alpha_outer=0,
    )
    temp_image = Image.new(mode="L", size=observed_cap.dim, color=255)
    observed_cap.generate_image(temp_image, is_annotate=False)
    _r1, _l1 = model.predict(*flat_chain.mean(axis=0), *inputs[r_idx])
    _x_coords, _y_coords = get_outline(_l1, _r1, observed_cap)
    plt.plot(_x_coords, _y_coords, alpha=0.5, color='blue', label='Mean Predicted')

    plt.imshow(temp_image, cmap="gray", aspect="auto")
    plt.legend()

    if save_path:
        plt.savefig(save_path / f'mean_shape_{r_idx}.png')

    plt.show()


def plot_trace(sampler: emcee.EnsembleSampler,
               model: BayesModel,
               save_path: Path = None):
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)

    samples = sampler.get_chain()
    labels = model.param_names

    for i in range(model.n_params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    if save_path:
        plt.savefig(save_path / 'traceplot.png')

    plt.show()


def plot_corner(sampler: emcee.EnsembleSampler,
                model: BayesModel,
                truths: list = None,
                save_path: Path = None):
    flat_samples = get_samples(sampler, model)
    bins = 20
    nsamples = flat_samples.shape[0]

    fig = corner.corner(flat_samples, bins=bins, labels=model.param_names, truths=truths)

    for i, dist in enumerate(model.param_dists):
        ax = fig.axes[i * model.n_params + i]
        params = np.linspace(*ax.get_xlim(), 100)
        params_space = params[1] - params[0]
        space_prob = dist.cdf(params + 0.5 * params_space) - dist.cdf(params - 0.5 * params_space)
        probs = nsamples * (100 / bins) * space_prob
        ax.plot(params, probs, 'b--')

    if save_path:
        plt.savefig(save_path / 'cornerplot.png')

    plt.show()


def plot_g_k_uncertainty(params: list,
                         inputs,
                         obs,
                         flat_chain,
                         save_path: Path = None, num_samples=300):
    true_G, true_K, true_p_factor = params
    x_rl = inputs[:, :2]
    fig, axes = plt.subplots(ncols=2, figsize=(13, 6))
    eps_z, eps_r = get_eps(inputs, obs)
    p_wall = inputs[:, 2] * true_p_factor
    pressure = inputs[:, 2]
    del_p = (p_wall - pressure)
    p_avg = 1 / 3 * (2 * p_wall + pressure)
    mean_g = get_line_param(2 * (eps_r - eps_z), del_p)
    mean_k = get_line_param(2 * eps_r + eps_z, p_avg)

    rand_idxs = rng.choice(len(flat_chain), size=num_samples, replace=False)
    flat_chain = flat_chain[rand_idxs, :]

    _x_G = np.linspace(0, 2, 10)
    _x_K = np.linspace(0, 0.6, 10)
    exponent = r"$\times 10^2$"
    axes.flat[0].plot(_x_G, _x_G * true_G, ls='--', color='black', label=f'G = {true_G:.3f}{exponent} kPa')
    axes.flat[0].legend()
    axes.flat[1].plot(_x_K, _x_K * true_K, ls='--', color='black', label=f'K = {true_K:.3f}{exponent} kPa')
    axes.flat[1].legend()
    # plot G from our obs
    axes.flat[0].scatter(2 * (eps_r - eps_z), mean_g(2 * (eps_r - eps_z)), color='red', label=f'Observed')
    # plot K from our obs
    axes.flat[1].scatter(2 * eps_r + eps_z, mean_k(2 * eps_r + eps_z), color='red', label=f'Observed')
    # plot G and K from our posterior
    for sample in flat_chain:
        sample_G, sample_K = sample[0], sample[1]
        axes.flat[0].plot(_x_G, _x_G * sample_G, ls='-', color='blue', alpha=0.01)
        axes.flat[1].plot(_x_K, _x_K * sample_K, ls='-', color='blue', alpha=0.01)

    axes.flat[0].set_xlabel(r'$2(\epsilon_r - \epsilon_z)$')
    axes.flat[0].set_ylabel(f'Min. wall pressure {exponent} kPa')
    axes.flat[1].set_xlabel(r'$2\epsilon_r + \epsilon_z$')
    axes.flat[1].set_ylabel(f'Avg. Pressure {exponent} kPa')

    axes.flat[0].legend()
    axes.flat[1].legend()
    #

    if save_path:
        plt.savefig(save_path / 'uncertainty_g_k.png')
    plt.show()
