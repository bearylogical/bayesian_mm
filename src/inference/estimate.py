from pathlib import Path
from time import strftime, time
import logging
from multiprocessing import Pool
import scipy.stats as stats
import numpy as np
import emcee
from src.inference.models.model import IsotropicModel, BayesModel
from src.utils.transforms import normalise_bands
from src.utils.utilities import set_logger

logger = set_logger()
from src.utils.shapes.capillary import CapillaryImage

import matplotlib.pyplot as plt
import corner

isotropic = IsotropicModel(noise=1,
                           G=stats.norm(loc=11, scale=5),
                           K=stats.norm(loc=6, scale=5),
                           p_factor=stats.norm(loc=1.5, scale=.5))

isotropic_2 = IsotropicModel(noise=1,
                             G=stats.norm(loc=1.1e4, scale=5e3),
                             K=stats.norm(loc=6e4, scale=5e3),
                             p_factor=stats.norm(loc=1.4, scale=.3))


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
    acts = sampler.get_autocorr_time()
    for param_name, act in zip(model.param_names, acts):
        logger.info('auto-correlation ({}): {}'.format(param_name, act))

    # burn-in and thinning
    act_max = int(np.max(acts))
    flat_samples = sampler.get_chain(discard=2 * act_max, thin=act_max // 2, flat=True)
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
        plt.savefig(save_path  / 'cornerplot.png')

    plt.show()
    return flat_samples


def run_experiment(model, num_synthetic, custom_experiment_name=None):
    rl_0 = np.array([117.9, 313.0])
    img_size = (2880, 2048)
    n_rl_0 = normalise_bands(rl_0, img_size=img_size)
    # observed = np.array([95.014, 390.641])
    # p_0 = 11220.0

    length_scale = 1e3
    # constants
    true_G = 10024.9056 / length_scale
    true_K = 60996.0301 / length_scale
    true_p_factor = 1.77776
    logger.info('Generating Data')
    pressures = np.linspace(1e3, 3e4, num_synthetic) / length_scale
    pressures = np.expand_dims(pressures, 1)
    # define our inputs
    initial_bands = np.expand_dims(n_rl_0, 0)
    # initial_bands_noise = initial_bands + np.array([1e-2, 1e-2]) * np.random.randn(2).T
    initial_bands = np.repeat(initial_bands, num_synthetic, axis=0)
    x = np.append(initial_bands, pressures, axis=1)

    y = isotropic.predict([true_G, true_K, true_p_factor], x)
    y_noise = y + np.array([1e-3, 1e-3]) * np.random.randn(2, num_synthetic).T


    logger.info('Starting Sampling')
    N_walkers = 100
    # get starting points
    start_pos = model.sample_prior(N_walkers)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(N_walkers, model.n_params, model.log_posterior, args=(x, y_noise), pool=pool)
        start = time()
        sampler.run_mcmc(start_pos, 5000, progress=True)
        end = time()
        multi_time = end - start
        logger.info("Sampling took {0:.1f} seconds".format(multi_time))

    temp_dir = Path.cwd() / 'estimation' / strftime("%Y%m%d")
    experiment_name = strftime("%Y%m%d_%H%M") if custom_experiment_name is None else custom_experiment_name
    temp_path = temp_dir / experiment_name
    temp_path.mkdir(parents=True, exist_ok=False)
    logger.info(f'Directory {temp_path} created, all plots will be saved here.')

    logger.info('Generating Trace plot')
    plot_trace(sampler, model, temp_path)
    logger.info('Generating Cornerplot')
    samples = plot_corner(sampler, model, truths=[true_G, true_K, true_p_factor], save_path=temp_path)

    return samples

def main():
    model = isotropic
    custom_experiment_name = "Increasing data"

    N_data = [1, 5, 10, 30]
    data = []
    for N in N_data:
        data.append(run_experiment(model, N, custom_experiment_name))



if __name__ == "__main__":
    main()
