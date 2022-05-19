import os

import scipy.stats as stats
import numpy as np
from time import strftime, time

from pathlib import Path

from src.inference.estimate import get_samples, log_results
from src.inference.plots import plot_r_l_scatter, plot_g_k_bands, plot_trace, plot_corner, plot_chain_obs, \
    plot_g_k_uncertainty, plot_capillary_observations, plot_g_k_uncertainty_obs, plot_posterior
from src.utils.mechanics import CapillaryStressBalance
from src.utils.transforms import normalise_bands
from src.utils.utilities import set_logger
from src.inference.models.model import IsotropicModel, IsotropicModelV2, IsotropicModelV3

logger = set_logger()
SEED = 42
rng = np.random.default_rng(SEED)

isotropic_normal = IsotropicModel(noise=1,
                                  G=stats.norm(loc=0, scale=1),
                                  K=stats.norm(loc=1, scale=2),
                                  p_factor=stats.uniform(loc=.5, scale=2))

isotropic_uniform = IsotropicModel(noise=1,
                                   G=stats.uniform(loc=.0, scale=.8),
                                   K=stats.uniform(loc=1, scale=5),
                                   p_factor=stats.uniform(loc=.5, scale=4))

isotropic_uniform_2 = IsotropicModel(noise=1,
                                     length_scale=1e2,
                                     G=stats.uniform(loc=0., scale=1),
                                     K=stats.uniform(loc=.1, scale=1),
                                     p_factor=stats.uniform(loc=1, scale=1))

isotropic_lognorm = IsotropicModel(noise=1,
                                   G=stats.lognorm(s=1.46, scale=1.06),
                                   K=stats.lognorm(s=1.46, scale=1.06),
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
    pressures = np.linspace(1e3, 1e4, num_obs) / kwargs.get("length_scale", 1e3)
    pressures = np.expand_dims(pressures, 1)
    # define our inputs
    initial_bands = np.expand_dims(initial_bands, 0)
    # in the case of fully isotropic behaviour, we assume that the deformation in the coordinate axes is uniform

    initial_bands = np.repeat(initial_bands, num_obs, axis=0)

    x = np.append(initial_bands, pressures, axis=1)
    x_experiments = np.repeat(x, m_observations, axis=0)
    y_true = model([G, K, p_factor], x_experiments)
    y_experiments = y_true + np.array([1e-3, 1e-3]) * rng.normal(size=(2, num_obs * m_observations)).T

    return x_experiments, y_true, y_experiments


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
    x, y, y_noise = generate_data_v4(num_obs=num_obs, m_observations=num_experiment, **material_params)
    # model.log_likelihood([0.30968005, 6.0420871,  4.40487208], x, y)
    likelihood_params = model.max_likelihood(x, y)
    model.log_likelihood(likelihood_params, x, y)
    logger.info(f"Max likelihood: {model.max_likelihood(x, y)}")
    logger.info('Starting Sampling')
    n_walkers = 20

    tstamp = strftime("%Y%m%d")
    dir_name = tstamp if custom_experiment_name is None else f"{custom_experiment_name}_{tstamp}"
    temp_dir = Path.cwd() / 'estimation' / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory {temp_dir} created, all plots will be saved here.')

    # get starting points
    sampler = model.sample(x, y_obs=y_noise, n_walkers=n_walkers, save_dir=temp_dir, save_chain=False)
    samples, _ = get_samples(sampler, model)

    log_results(samples, ["G", "K", "p_factor"])
    # data = pd.DataFrame(columns=["G", "K", "p_factor"], data=samples)
    # data["num_obs_per_exp"] =  num_obs
    # data["num_exp"] = num_experiment
    # return data

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
                    num_experiment: int = 1,
                    custom_experiment_name=None,
                    **kwargs):
    script_start = time()
    sb = CapillaryStressBalance()
    bands = sb.get_bands(x_coords, y_coords)
    bands = normalise_bands(bands)
    _, _, alpha = sb.process_particle_points(x_coords, y_coords)
    v_strain, eps_g, wall_min_pressures, avg_pressures = sb.calculate(x_coords, y_coords, pressures)

    n_rl_0, y_obs = bands[0, :], bands[1:, :]
    initial_bands = np.expand_dims(n_rl_0, 0)
    # in the case of fully isotropic behaviour, we assume that the deformation in the coordinate axes is uniform
    #
    initial_bands = np.repeat(initial_bands, len(y_obs), axis=0)
    pressures = np.expand_dims(pressures, 1) / kwargs.get("length_scale", 1e3)
    x = np.append(initial_bands, pressures[1:len(y_obs) + 1], axis=1)
    x = np.repeat(x, num_experiment, axis=0)
    y_obs = np.repeat(y_obs, num_experiment, axis=0)
    y_obs = y_obs + np.array([1e-4, 1e-4]) * np.random.randn(2, len(y_obs)).T
    logger.info(f"Max likelihood: {model.max_likelihood(x, y_obs)}")
    logger.info('Starting Sampling')
    n_walkers = 25

    sampler = model.sample(x, y_obs=y_obs, n_walkers=n_walkers)
    samples, _ = get_samples(sampler, model)
    labels = ["G", "K", "p_factor"]

    log_results(samples, labels)

    tstamp = strftime("%Y%m%d")
    dir_name = tstamp if custom_experiment_name is None else f"{custom_experiment_name}_{tstamp}"
    temp_dir = Path.cwd() / 'estimation' / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory {temp_dir} created, all plots will be saved here.')

    logger.info('Saving plot of generated data, with first experiment illustrated')
    plot_capillary_observations(n_rl_0, y_obs[0::num_experiment], alpha=alpha,
                                figsize=(16, 8), save_path=temp_dir)

    logger.info('Generating Trace plot')
    plot_trace(sampler, model, temp_dir)
    logger.info('Generating Corner plot')
    plot_corner(sampler, model, save_path=temp_dir)

    logger.info('Generating HDI plot')
    plot_posterior(sampler, labels, save_path=temp_dir)

    logger.info('Saving scatter and band plots')
    plot_r_l_scatter(y_obs, m_observations=num_experiment, save_path=temp_dir)

    logger.info('Generating sample overlays')
    plot_chain_obs(x, y_obs, samples, alpha=alpha, model=model, save_path=temp_dir)

    logger.info('Plotting uncertainty plot')
    plot_g_k_uncertainty_obs(obs_x_coord=x_coords,
                             obs_y_coord=y_coords,
                             pressures=pressures.flatten(),
                             flat_chain=samples,
                             save_path=temp_dir)
    scipt_end = time()
    logger.info(f"Process took {scipt_end - script_start:.1f} seconds")


def check_likelihood():
    num_obs = 8
    num_experiment = 3
    rl_0 = np.array([117.9, 313.0])
    img_size = (2880, 2048)
    alpha = 0.06
    # rl_0 = np.expand_dims(rl_0, 0)
    n_rl_0 = normalise_bands(rl_0, img_size=img_size)
    # observed = np.array([95.014, 390.641])
    # p_0 = 11220.0
    length_scale = 1
    material_params = {
        "G": 10024.9056 / length_scale,
        "K": 60996.0301 / length_scale,
        "p_factor": 1.77776,
        "length_scale": length_scale
    }
    data = generate_data(n_rl_0, IsotropicModel(), num_obs=num_obs, m_observations=num_experiment, **material_params)

    model = IsotropicModel(noise=1,
                           length_scale=100,
                           G=stats.uniform(loc=.0, scale=1),
                           K=stats.uniform(loc=.0, scale=1),
                           p_factor=stats.uniform(loc=.5, scale=4))
    model.log_likelihood(list(material_params.values())[:3], data[0], data[1])

    model.max_likelihood(data[0], data[1])

def generate_data_v2(num_obs: int = 5, num_experiments: int = 1,model = IsotropicModelV2(),
                  **kwargs):
    G, K = kwargs.get("G"), kwargs.get("K")
    # define our inputs
    eps_strains = np.linspace(0, 0.5, num_obs)
    eps_strains = np.stack((eps_strains, eps_strains))

    x_experiments = np.repeat(eps_strains, num_experiments, axis=0)
    y_true = model([G, K], x_experiments)
    y_experiments = y_true + np.array([1e-3, 1e-3]) * np.random.normal(size=(num_obs * num_experiments, 2))

    return x_experiments, y_true, y_experiments

def generate_data_v3(num_obs: int = 5, num_experiments: int = 1,model = IsotropicModelV3(),
                  **kwargs):
    G = kwargs.get("G")
    # define our inputs
    eps_strains = np.linspace(0, 0.5, num_obs)

    x_experiments = np.repeat(eps_strains, num_experiments, axis=0)
    y_true = model(G, x_experiments).T
    y_experiments = y_true + np.array([1e-3]) * np.random.normal(size=(num_obs * num_experiments, 1))

    return x_experiments, y_true, y_experiments


def generate_data_v4(num_obs: int = 5, num_experiments: int = 1,model = IsotropicModel(),
                  **kwargs):
    G, K, p_factor = kwargs.get("G"), kwargs.get("K"), kwargs.get("p_factor")
    # define our inputs
    eps_strains = np.linspace(0, 0.5, num_obs)
    eps_strains = np.stack((eps_strains, eps_strains), axis=1)

    pressures = np.linspace(1e3, 1e4, num_obs) / kwargs.get("length_scale", 1e3)
    pressures = np.expand_dims(pressures, 1)

    x = np.append(eps_strains, pressures, axis=1)
    x_experiments = np.repeat(x, num_experiments, axis=0)

    y_true = model([G, K, p_factor], x_experiments)
    y_experiments = y_true + np.array([1e-3, 1e-3]) * np.random.normal(size=(num_obs * num_experiments, 2))

    return x_experiments, y_true, y_experiments


def run_experimentv2(model, num_obs, num_experiment, custom_experiment_name=None):
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
        "length_scale": length_scale
    }
    logger.info('Generating Data')
    x, y, y_noise = generate_data_v2(num_obs=num_obs, m_observations=num_experiment, **material_params)
    # model.log_likelihood([0.30968005, 6.0420871,  4.40487208], x, y)
    likelihood_params = model.max_likelihood(x, y)
    model.log_likelihood(likelihood_params, x, y)
    logger.info(f"Max likelihood: {model.max_likelihood(x, y)}")
    logger.info('Starting Sampling')
    n_walkers = 20

    tstamp = strftime("%Y%m%d")
    dir_name = tstamp if custom_experiment_name is None else f"{custom_experiment_name}_{tstamp}"
    temp_dir = Path.cwd() / 'estimation' / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory {temp_dir} created, all plots will be saved here.')

    # get starting points
    sampler = model.sample(x, y_obs=y_noise, n_walkers=n_walkers, save_dir=temp_dir, save_chain=False)
    samples, _ = get_samples(sampler, model)

    log_results(samples, ["G", "K", "p_factor"])
    # data = pd.DataFrame(columns=["G", "K", "p_factor"], data=samples)
    # data["num_obs_per_exp"] =  num_obs
    # data["num_exp"] = num_experiment
    # return data

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

def main():
    model = isotropic_uniform
    custom_experiment_name = "isotropic_uniform_5_3"
    exp_dir = "dataset/Inc_press_2"
    kp_file = "dataset/Inc_press_2/Inc_press_2.txt"
    pressure_file = "dataset/Inc_press_2/Pressure_mm_H20_dummy.txt"
    num_obs, num_experiments = 5, 3
    # x_points, y_points, pressures = load_keypoints_pressures(kp_file, pressure_file, normalize=True)
    run_experiment(model, num_obs, num_experiments, custom_experiment_name)


if __name__ == "__main__":
    main()
    # a, b, c = generate_data_v3(G=.6,)
    # a, b, c = generate_data_v2(G=.1, K=.7)
    # new_model = IsotropicModelV2(noise=1,
    #                              length_scale=100,
    #                              G=stats.uniform(loc=.0, scale=1),
    #                              K=stats.uniform(loc=.0, scale=1))
    # new_model.max_likelihood(a, b)