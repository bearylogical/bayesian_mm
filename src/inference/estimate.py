from pathlib import Path
from time import strftime, time
import scipy.stats as stats
import numpy as np
import emcee
from src.inference.models.model import IsotropicModel, BayesModel
from src.inference.plots import plot_r_l_scatter, plot_g_k_bands, plot_trace, plot_corner, plot_chain_obs, \
    plot_g_k_uncertainty, plot_capillary_observations, get_samples, plot_g_k_uncertainty_obs
from src.processors.imagej import load_keypoints_pressures
from src.utils.mechanics import CapillaryStressBalance
from src.utils.transforms import normalise_bands
from src.utils.utilities import set_logger

logger = set_logger()

isotropic_normal = IsotropicModel(noise=1,
                                  G=stats.norm(loc=.2, scale=.05),
                                  K=stats.norm(loc=.7, scale=.3),
                                  p_factor=stats.uniform(loc=.5, scale=4))

isotropic_uniform = IsotropicModel(noise=1,
                                   G=stats.uniform(loc=.0, scale=.8),
                                   K=stats.uniform(loc=1, scale=5),
                                   p_factor=stats.uniform(loc=.5, scale=4),
                                   length_scale=1e4)

isotropic_lognorm = IsotropicModel(noise=1,
                                   G=stats.lognorm(s=0.24622, scale=-1.63975),
                                   K=stats.lognorm(s=0.410636, scale=-0.440986),
                                   p_factor=stats.uniform(loc=.5, scale=4))

isotropic_2 = IsotropicModel(noise=1,
                             G=stats.norm(loc=1.1e4, scale=5e3),
                             K=stats.norm(loc=6e4, scale=5e3),
                             p_factor=stats.norm(loc=0.5, scale=4))


def generate_data(initial_bands: np.ndarray, model: IsotropicModel, num_obs: int = 10, m_observations: int = 1,
                  **kwargs):
    if len(initial_bands) != 2:
        raise Exception('Band must only be of dim 2')

    G, K, p_factor = kwargs.get("G"), kwargs.get("K"), kwargs.get("p_factor")
    pressures = np.linspace(1e3, 3e4, num_obs) / kwargs.get("length_scale", 1e3)
    pressures = np.expand_dims(pressures, 1)
    # define our inputs
    initial_bands = np.expand_dims(initial_bands, 0)
    # in the case of fully isotropic behaviour, we assume that the deformation in the coordinate axes is uniform
    #
    initial_bands = np.repeat(initial_bands, num_obs, axis=0)

    x = np.append(initial_bands, pressures, axis=1)
    x_experiments = np.repeat(x, m_observations, axis=0)

    y_true = model.predict([G, K, p_factor], x_experiments)
    y_experiments = y_true + np.array([1e-3, 1e-2]) * np.random.randn(2, num_obs * m_observations).T

    return x_experiments, y_true, y_experiments


def log_results(samples, labels: list, percentiles: list = (16, 50, 84)):
    ndim = samples.ndim

    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], percentiles)
        q = np.diff(mcmc)
        logger.info(f"Sampled {labels[i]} = {mcmc[1]:.3f} - {q[0]:.3f} / +{q[1]:.3f}")


def run_experiment(model, num_obs, num_experiment, custom_experiment_name=None):
    rl_0 = np.array([117.9, 313.0])
    img_size = (2880, 2048)
    alpha = 0.06
    # rl_0 = np.expand_dims(rl_0, 0)
    n_rl_0 = normalise_bands(rl_0, img_size=img_size)
    # observed = np.array([95.014, 390.641])
    # p_0 = 11220.0
    script_start = time()
    length_scale = 1e5
    material_params = {
        "G": 10024.9056 / length_scale,
        "K": 60996.0301 / length_scale,
        "p_factor": 1.77776,
        "length_scale": length_scale
    }
    logger.info('Generating Data')
    x, y, y_noise = generate_data(n_rl_0, model, num_obs=num_obs, m_observations=num_experiment, **material_params)

    logger.info('Starting Sampling')
    N_walkers = 50
    # get starting points
    start_pos = model.sample_prior(N_walkers)

    sampler = emcee.EnsembleSampler(N_walkers, model.n_params, model.log_posterior, args=(x, y_noise))
    start = time()
    sampler.run_mcmc(start_pos, 5000, progress=True)
    end = time()
    multi_time = end - start
    logger.info("Sampling took {0:.1f} seconds".format(multi_time))
    samples = get_samples(sampler, model)
    logger.info(f"Max likelihood: {model.max_likelihood(x, y_noise)}")
    log_results(samples, ["G", "K", "p_factor"])

    tstamp = strftime("%Y%m%d")
    dir_name = tstamp if custom_experiment_name is None else f"{custom_experiment_name}_{tstamp}"
    temp_dir = Path.cwd() / 'estimation' / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory {temp_dir} created, all plots will be saved here.')

    logger.info('Saving plot of generated data, with first experiment illustrated')
    plot_capillary_observations(n_rl_0, y[0::num_experiment], noise_obs=y_noise[0::num_experiment], alpha=alpha,
                                figsize=(16, 8), save_path=temp_dir)

    logger.info('Saving scatter and band plots')
    plot_r_l_scatter(y_noise, m_observations=num_experiment, save_path=temp_dir)
    plot_g_k_bands(x, y_noise, material_params["p_factor"], m_observations=num_experiment, save_path=temp_dir)

    logger.info('Generating Trace plot')
    plot_trace(sampler, model, temp_dir)
    logger.info('Generating Corner plot')
    plot_corner(sampler, model, truths=list(material_params.values())[:3], save_path=temp_dir)

    logger.info('Generating sample overlays')
    plot_chain_obs(x, y_noise, samples, alpha=alpha, model=model, save_path=temp_dir)

    logger.info('Plotting uncertainty plot')
    plot_g_k_uncertainty(params=[material_params[k] for k in ["G", "K", "p_factor"]],
                         inputs=x,
                         obs=y_noise,
                         flat_chain=samples,
                         save_path=temp_dir)
    scipt_end = time()
    logger.info(f"Process took {scipt_end - script_start:.1f} seconds")

    return samples


def estimate_params(x_coords,
                    y_coords,
                    pressures,
                    model,
                    num_experiment:int=1,
                    custom_experiment_name=None,
                    **kwargs):

    script_start = time()
    sb = CapillaryStressBalance()
    bands = sb.get_bands(x_coords, y_coords)
    bands = normalise_bands(bands)
    _, _, alpha = sb.process_particle_points(x_coords, y_coords)

    n_rl_0, y_obs = bands.T[0, :], bands.T[1:, :]
    initial_bands = np.expand_dims(n_rl_0, 0)
    # in the case of fully isotropic behaviour, we assume that the deformation in the coordinate axes is uniform
    #
    initial_bands = np.repeat(initial_bands, len(y_obs), axis=0)
    pressures = np.expand_dims(pressures, 1) / kwargs.get("length_scale", 1e3)
    x = np.append(initial_bands, pressures[1:len(y_obs)+1], axis=1)
    x = np.repeat(x, num_experiment, axis=0)
    y_obs = np.repeat(y_obs, num_experiment, axis=0)
    # y_obs = y_obs + np.array([1e-4, 1e-4]) * np.random.randn(2, len(y_obs)).T
    logger.info(f"Max likelihood: {model.max_likelihood(x, y_obs)}")
    logger.info('Starting Sampling')
    N_walkers = 50
    # get starting points
    start_pos = model.sample_prior(N_walkers)

    sampler = emcee.EnsembleSampler(N_walkers, model.n_params, model.log_posterior, args=(x, y_obs))
    start = time()
    sampler.run_mcmc(start_pos, 5000, progress=True)
    end = time()
    multi_time = end - start
    logger.info("Sampling took {0:.1f} seconds".format(multi_time))
    samples = get_samples(sampler, model)

    log_results(samples, ["G", "K", "p_factor"])
    #
    tstamp = strftime("%Y%m%d")
    dir_name = tstamp if custom_experiment_name is None else f"{custom_experiment_name}_{tstamp}"
    temp_dir = Path.cwd() / 'estimation' / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory {temp_dir} created, all plots will be saved here.')

    logger.info('Saving plot of generated data, with first experiment illustrated')
    plot_capillary_observations(n_rl_0,  y_obs[0::num_experiment], alpha=alpha,
                                figsize=(16, 8), save_path=temp_dir)

    logger.info('Generating Trace plot')
    plot_trace(sampler, model, temp_dir)
    logger.info('Generating Corner plot')
    plot_corner(sampler, model, save_path=temp_dir)

    logger.info('Saving scatter and band plots')
    plot_r_l_scatter(y_obs, m_observations=num_experiment, save_path=temp_dir)

    logger.info('Generating sample overlays')
    plot_chain_obs(x, y_obs, samples, alpha=alpha, model=model, save_path=temp_dir)

    logger.info('Plotting uncertainty plot')
    plot_g_k_uncertainty_obs(obs_x_coord=y_obs[:,0],
                             obs_y_coord=y_obs[:,1],
                             pressures=pressures,
                         flat_chain=samples,
                         save_path=temp_dir)
    scipt_end = time()
    logger.info(f"Process took {scipt_end - script_start:.1f} seconds")


def main():
    model = isotropic_uniform
    custom_experiment_name = "isotropic_uniform_5exp"
    exp_dir = "dataset/Inc_press_2"
    kp_file = "dataset/Inc_press_2/Inc_press_2.txt"
    pressure_file = "dataset/Inc_press_2/Pressure_mm_H20_dummy.txt"
    num_obs, num_experiments = 5, 5
    x_points, y_points, pressures = load_keypoints_pressures(kp_file, pressure_file, normalize=False)
    run_experiment(model, num_obs, num_experiments, custom_experiment_name)
    #
    # estimate_params(x_points, y_points, pressures, model,
    #                 num_experiment=num_experiments,
    #                 custom_experiment_name=custom_experiment_name, length_scale=1e5)
    # N_data = [1, 5, 10, 30]
    # data = []
    # for N in N_data:
    #     data.append(run_experiment(model, N, custom_experiment_name))


if __name__ == "__main__":
    main()
