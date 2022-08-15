from typing import Tuple

import numpy as np
import emcee
from scipy import stats as stats

from src.inference.evaluate import get_psrf

import logging

logger = logging.getLogger("bayesian_nn")


def get_max_aposteriori(sampler: emcee.EnsembleSampler):
    sampler.get_log_prob()


def get_max_n_burn(
    sampler: emcee.EnsembleSampler,
    target_psrf: float = 1.1,
    n_start: int = 2,
    n_burn: int = 10,
):

    n_samples, _, n_dims = sampler.get_chain().shape
    psrfs = np.zeros(shape=(n_samples, n_dims))
    for i in range(n_start, n_samples):
        psrfs[i] = get_psrf(sampler, i)
    logger.info(f"PSRF at end of chain: {psrfs[-1]}")
    for dim in range(n_dims):
        idx_psrf_min = np.argwhere(psrfs[n_start:, dim] < target_psrf).flat[0]
        if n_burn <= idx_psrf_min:
            n_burn = idx_psrf_min

    return n_burn


def get_sample_uncorrelated(sampler: emcee.EnsembleSampler):
    acts = sampler.get_autocorr_time()
    # burn-in and thinning
    act_max = int(np.max(acts))
    burn_in = get_max_n_burn(sampler)
    thin = act_max
    logger.info(f"Burn-in: {burn_in}")
    logger.info(f"Thin: {thin}")

    return burn_in, thin


# TODO : this should be a class
def get_samples(
    sampler: emcee.EnsembleSampler, labels: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    acts = sampler.get_autocorr_time()
    if labels is not None:
        for param_name, act in zip(labels, acts):
            logger.info("auto-correlation ({}): {}".format(param_name, act))
    # TODO: struct for chain
    burn_in, thin = get_sample_uncorrelated(sampler)

    chain_params = dict(discard=burn_in, thin=thin, flat=True)

    flat_samples: np.ndarray = sampler.get_chain(**chain_params)
    log_prob_samples: np.ndarray = sampler.get_log_prob(**chain_params)
    logger.info(f"Flat chain shape: {flat_samples.shape}")
    logger.info(f"Log prob shape: {log_prob_samples.shape}")
    blobs = sampler.get_blobs(**chain_params)

    log_prior_samples, log_likelihood_samples = blobs["log_prior"], blobs["log_ll"]
    logger.info(f"Log prior shape: {log_prior_samples.shape}")
    logger.info(f"Log likelihood shape: {log_likelihood_samples.shape}")

    return flat_samples, log_prob_samples, log_prior_samples, log_likelihood_samples


# def log_results(samples: np.ndarray, labels: list, percentiles: list = (16, 50, 84)):
#     ndim = samples.ndim

#     for i in range(ndim):
#         mcmc = np.percentile(samples[:, i], percentiles)
#         max_ap = stats.mode(samples[:, i])
#         q = np.diff(mcmc)
#         logger.info(
#             f"Sampled {labels[i]} = MAP: {max_ap}, {mcmc[1]:.3f} - {q[0]:.3f} / +{q[1]:.3f}"
#         )
