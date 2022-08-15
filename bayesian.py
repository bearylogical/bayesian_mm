import os

import scipy.stats as stats
import numpy as np
from emcee.moves import GaussianMove
from time import strftime, time

from pathlib import Path
from src.inference.plots import plot_results

# from src.inference.estimate import get_samples, log_results
# from src.inference.plots import (
#     plot_r_l_scatter,
#     plot_g_k_bands,
#     plot_trace,
#     plot_corner,
#     plot_chain_obs,
#     plot_g_k_uncertainty,
#     plot_capillary_observations,
#     plot_g_k_uncertainty_obs,
#     plot_posterior,
# # )
# from src.utils.mechanics import CapillaryStressBalance
from src.utils.transforms import normalise_bands
from src.utils.utilities import set_logger
from src.inference.sampler import BaseSampler, BayesSampler, default_model

logger = set_logger()
SEED = 42
rng = np.random.default_rng(SEED)


# def generate_data(
#     initial_bands: np.ndarray,
#     model: IsotropicModel,
#     num_obs: int = 10,
#     m_observations: int = 1,
#     **kwargs,
# ):
#     if len(initial_bands) != 2:
#         raise Exception("Band must only be of dim 2")

#     G, K, p_factor = kwargs.get("G"), kwargs.get("K"), kwargs.get("p_factor")
#     pressures = np.linspace(1e3, 1e4, num_obs) / kwargs.get("length_scale", 1e3)
#     pressures = np.expand_dims(pressures, 1)
#     # define our inputs
#     initial_bands = np.expand_dims(initial_bands, 0)
#     # in the case of fully isotropic behaviour, we assume that the deformation in the coordinate axes is uniform

#     initial_bands = np.repeat(initial_bands, num_obs, axis=0)

#     x = np.append(initial_bands, pressures, axis=1)
#     x_experiments = np.repeat(x, m_observations, axis=0)
#     y_true = model([G, K, p_factor], x_experiments)
#     y_experiments = (
#         y_true
#         + np.array([1e-3, 1e-3]) * rng.normal(size=(2, num_obs * m_observations)).T
#     )

#     return x_experiments, y_true, y_experiments
# priors = dict(
#     G=stats.lognorm(s=1, scale=30),
#     K=stats.lognorm(s=1, scale=30),
#     noise_1=1,
#     noise_2=1,
# )
priors = dict(
    G=stats.norm(loc=30, scale=10),
    K=stats.norm(loc=30, scale=10),
    noise_1=stats.uniform(0.01, 5),
    noise_2=stats.uniform(0.01, 5),
)
default_sampler = BayesSampler(priors, model_fn=default_model)


def run_experiment(
    num_obs,
    num_experiment,
    custom_experiment_name=None,
    sampler: BaseSampler = default_sampler,
):
    script_start = time()
    length_scale = 1e3
    material_params = {
        "G": 30_000 / length_scale,
        "K": 30_000 / length_scale,
        "p_factor": 1.77776,
        "length_scale": length_scale,
    }
    logger.info("Generating Data")
    x, y, y_noise = generate_data_v2(
        num_obs=num_obs, m_observations=num_experiment, **material_params
    )
    # model.log_likelihood([0.30968005, 6.0420871,  4.40487208], x, y)
    # likelihood_params = model.max_likelihood(x, y)
    # model.log_likelihood(likelihood_params, x, y)
    logger.info(f"Max likelihood: {sampler.max_likelihood(x, y)}")
    logger.info("Starting Sampling")
    num_walkers = 15
    num_chains = 10_000

    tstamp = strftime("%Y%m%d")
    dir_name = (
        tstamp
        if custom_experiment_name is None
        else f"{custom_experiment_name}_{tstamp}"
    )
    temp_dir = Path.cwd() / "estimation" / dir_name
    temp_dir = temp_dir / strftime("%Y%m%d_%H%M")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory {temp_dir} created, all plots will be saved here.")

    # get starting points
    sampler.fit(
        x,
        y_noise,
        num_walkers=num_walkers,
        num_chains=num_chains,
        save_dir=temp_dir,
        moves=None,
        save_chain=False,
    )
    sampler.get_samples()
    plot_results(x, y_noise, sampler)

    scipt_end = time()
    logger.info(f"Process took {scipt_end - script_start:.1f} seconds")

    # return samples


def generate_data_v2(
    num_obs: int = 5, num_experiments: int = 1, model=default_model, **kwargs
):
    G, K = kwargs.get("G"), kwargs.get("K")
    # define our inputs
    eps_strains_1 = np.linspace(0, 0.2, num_obs)
    eps_strains_2 = np.linspace(
        0, 0.6, num_obs
    )  # replicate what we see from real data, where G is more noticeable
    eps_strains = np.stack((eps_strains_1, eps_strains_2))

    x_experiments = np.repeat(eps_strains.T, num_experiments, axis=0)
    y_true = model([G, K], x_experiments)
    y_experiments = y_true + rng.normal(size=(num_obs * num_experiments, 2))

    return x_experiments, y_true, y_experiments


def main():
    custom_experiment_name = "isotropic_uniform_5_3"
    exp_dir = "dataset/Inc_press_2"
    kp_file = "dataset/Inc_press_2/Inc_press_2.txt"
    pressure_file = "dataset/Inc_press_2/Pressure_mm_H20_dummy.txt"
    num_obs, num_experiments = 5, 3
    # x_points, y_points, pressures = load_keypoints_pressures(kp_file, pressure_file, normalize=True)
    run_experiment(num_obs, num_experiments, custom_experiment_name)


if __name__ == "__main__":
    main()
