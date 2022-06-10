import os
from pathlib import Path
from typing import Union

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import emcee
import logging
from time import strftime, time

logger = logging.getLogger('bayesian_nn')

# fix the seed for the RNG
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


class BayesModel:
    """
    Base model class for Bayesian parameter inference
    """

    def __init__(self,
                 noise=1,
                 length_scale: float = 100,
                 sampler: emcee.EnsembleSampler = None,
                 **kwargs):
        self.noise = noise
        self.params = kwargs
        self.param_dists = list(self.params.values())
        self.param_names = list(self.params.keys())
        self.n_params = len(self.params)
        self.length_scale = length_scale
        self.sampler = sampler
        self.samples = None

    def __str__(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return NotImplementedError

    def sample_prior(self, size):
        return np.array([dist.rvs(size=size) for dist in self.param_dists]).T

    def log_prior(self, params):
        """
        (summed) Logarithm of the prior probability object.

        Parameters
        ----------
        params

        Returns
        -------

        """
        return sum(dist.logpdf(param) for param, dist in zip(params, self.param_dists))

    def log_likelihood(self, params, inputs, obs):
        rl_model = self(params, inputs)
        return np.sum(stats.norm.logpdf(x=rl_model.T.flatten() - obs.T.flatten(), loc=0, scale=self.noise))

    def log_posterior(self, params, inputs, obs):
        """
        Calculate logarithm of the posterior

        Parameters
        ----------
        params
        inputs
        obs

        Returns
        -------
        log posterior,log posterior_predictive
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, -np.inf
        ll = self.log_likelihood(params, inputs, obs)
        if not np.isfinite(ll):
            return lp, -np.inf, -np.inf
        return lp + ll, lp, ll

    def max_likelihood(self, inputs, obs):
        """
        Compute the maximum likelihood estimate.

        Parameters
        ----------
        inputs
        obs

        Returns
        -------

        """
        init_guss = self.sample_prior(1)
        soln = minimize(lambda x: -self.log_likelihood(x, inputs, obs), init_guss)
        return soln.x

    def save(self, save_posterior: bool = False):
        """

        Parameters
        ----------
        save_posterior: flag to store the posterior to be used as the next prior.

        Returns
        -------

        """
        return NotImplementedError

    def sample(self, x, y_obs, **kwargs):
        n_walkers = kwargs.get("n_walkers", 50)
        start_pos = kwargs.get("start_pos", self.sample_prior(n_walkers))
        is_save = kwargs.get("save_chain", False)
        num_chains = kwargs.get("num_chains", 5000)

        if is_save:
            fp = kwargs.get("save_dir", Path(os.getcwd())) / "chain.h5"
            backend = emcee.backends.HDFBackend(str(fp))
            backend.reset(n_walkers, self.n_params)
        else:
            backend = None
        dtype = [("log_prior", float), ("log_ll", float)]
        sampler = emcee.EnsembleSampler(n_walkers,
                                        self.n_params,
                                        self.log_posterior,
                                        args=(x, y_obs),
                                        backend=backend,
                                        blobs_dtype=dtype)
        start = time()
        sampler.run_mcmc(start_pos, num_chains, progress=True)
        end = time()
        multi_time = end - start
        logger.info("Sampling took {0:.1f} seconds".format(multi_time))

        return sampler

    def load_prior(self):
        return NotImplementedError


class IsotropicModel(BayesModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        A = np.array([[1, -1],
                      [2, 1]])
        self.A_inv = np.linalg.inv(A)

    def __call__(self, params, inputs) -> np.ndarray:
        """

        Parameters
        ----------
        params : G, K, p_factor (in a list)
        inputs: rl_0 as a numpy array, p_0

        Returns
        -------
        Numpy (2x1) array with [[r] ,[l]]
        """
        G, K, p_factor = params
        p_0 = inputs

        p_diff = 1 / 2 * ((p_factor - 1) * p_0)
        p_avg = 1 / 3 * (p_0 * (2 * p_factor + 1))
        b1 = p_diff / G
        b2 = p_avg / K
        b = np.array([b1, b2])

        return np.stack([b1, b2, p_diff, p_avg], axis=1)


class IsotropicModelV2(BayesModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, params, inputs):
        """

        Parameters
        ----------
        params : G, K, p_factor (in a list)
        inputs: rl_0 as a numpy array, p_0

        Returnse
        -------
        Numpy (2x1) array with [[r] ,[l]]
        """
        G, K = params
        eps_g, eps_k = inputs

        p_avg = K * eps_k
        p_diff = G * eps_g

        return np.array([p_diff, p_avg]).T

class IsotropicModelV3(BayesModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_names = ["G", "K", "theta_G", "theta_K"]
        self.n_params = len(self.param_names)

    def log_prior(self, params):
        G, K, theta_G, theta_K = params
        if any([param < 0 for param in [G, K]]): # enforce param>0 condition
            return -np.inf
        else:
            return -1.5 * (np.log(2 + G ** 2 + K ** 2))

    def sample_prior(self, size):
        return np.array([np.random.random(size=size) for _ in range(self.n_params)]).T

    def __call__(self, params, inputs):
        """

        Parameters
        ----------
        params : G, K, p_factor (in a list)
        inputs: rl_0 as a numpy array, p_0

        Returnse
        -------
        Numpy (2x1) array with [[r] ,[l]]
        """
        G, K, theta_G, theta_K = params
        eps_g, eps_k = inputs

        p_avg = K * eps_k + theta_K
        p_diff = G * eps_g + theta_G

        return np.array([p_diff, p_avg]).T


class InvariantIsotropicModel(IsotropicModelV2):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_names = ["G", "K"]
        self.n_params = len(self.param_names)

    def log_prior(self, params):
        G, K = params # jeffey's prior for paramterisation of the mean.
        if G < 0 or K < 0:
            return -np.inf
        else:
            return 0.0
        # return -1.5 * (np.log(2 + G ** 2 + K ** 2))

    def sample_prior(self, size):
        return np.array([np.random.random(size=size) for _ in range(self.n_params)]).T


class InvariantIsotropicModelV2(InvariantIsotropicModel):
    def log_prior(self, params):
        G, K = params
        if G < 0 or K < 0:
            return -np.inf
        else:
            return -1.5 * (np.log(2 + G ** 2 + K ** 2))


class InvariantGeometricModelV3(IsotropicModelV2):

    def __call__(self, params, inputs):
        """

        Parameters
        ----------
        params : G, K, p_factor (in a list)
        inputs: rl_0 as a numpy array, p_0

        Returnse
        -------
        Numpy (2x1) array with [[r] ,[l]]
        """
        theta_G, theta_K = params
        eps_g, eps_k = inputs

        G, K = np.sin(theta_G), np.sin(theta_K)

        p_avg = K * eps_k
        p_diff = G * eps_g

        return np.array([p_diff, p_avg]).T

