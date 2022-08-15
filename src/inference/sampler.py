from abc import ABC, abstractmethod
from collections import namedtuple
import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
import scipy.stats as stats
import scipy.special as special
from scipy.optimize import minimize
import emcee
from emcee.moves import GaussianMove, Move
import logging
from time import time

from src.inference.priors import CustomPrior


from src.inference.estimate import get_samples as _get_samples

logger = logging.getLogger("bayesian_nn")

# fix the seed for the RNG
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


ChainSample = namedtuple("ChainSample", ("samples", "log_prob", "log_ll", "log_prior"))


def default_model(params: np.ndarray, inputs: np.ndarray):
    # mod_g, mod_k = params
    # strain_shear, strain_compre = inputs

    # stress_shear = mod_g * strain_shear
    # stress_compre = mod_k * strain_compre

    return inputs * params


def strain_model(params: np.ndarray, inputs: np.ndarray):
    # mod_g, mod_k = params
    # stress_shear, stress_compre = inputs

    # strain_shear = stress_shear / mod_g
    # strain_compre = stress_compre / mod_k

    return inputs / params


class BaseSampler(ABC):
    def __init__(self, parameters: dict, model_fn: Callable = default_model):
        self._params = parameters
        self._model_fn = model_fn
        self._sampler = None
        self._samples: ChainSample = None
        self._mle: list = None

    @property
    def model_fn(self):
        assert self._model_fn is not None
        return self._model_fn

    @property
    def noise(self):
        return {k: v for k, v in self.params.items() if "noise" in k}

    @property
    def params(self):
        return self._params

    @property
    def param_names(self) -> list:
        return [
            k
            for k, v in self.params.items()
            if isinstance(v, (stats.distributions.rv_frozen, CustomPrior))
        ]

    @property
    def param_dist(self) -> list:
        return [
            v
            for v in self.params.values()
            if isinstance(v, (stats.distributions.rv_frozen, CustomPrior))
        ]

    @property
    def n_params(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
        return len(self.param_dist)

    def sample_prior(self, size):
        return np.array([dist.rvs(size=size) for dist in self.param_dist]).T

    def prior(self, params):
        return np.array(
            [dist.pdf(param) for param, dist in zip(params, self.param_dist)]
        )

    def log_prior(self, params):
        """
        (summed) Logarithm of the prior probability object.

        Parameters
        ----------
        params

        Returns
        -------

        """
        return sum(dist.logpdf(param) for param, dist in zip(params, self.param_dist))

    @property
    def acceptance_fraction(self):
        assert isinstance(self.sampler, emcee.EnsembleSampler)
        print(self.sampler.acceptance_fraction)

    @abstractmethod
    def log_likelihood(self, *args):
        pass

    @abstractmethod
    def log_posterior(self):
        pass

    def get_chain(self, **kwargs):
        assert self.sampler is not None
        assert isinstance(self.sampler, emcee.EnsembleSampler)

        return self.sampler.get_chain(**kwargs)

    def get_samples(self):
        assert self.sampler is not None

        if self._samples is None:
            _samples, _log_prob, _log_prior, _log_ll = _get_samples(
                self.sampler, self.param_names
            )
            self._samples = ChainSample(_samples, _log_prob, _log_prior, _log_ll)

        return self._samples

    def max_likelihood(self, inputs, obs, errors: bool = True):
        init_guess = self.sample_prior(1)
        bounds = [(None, None), (None, None)]
        if len(self.param_dist) - len(self.noise) != 0:
            bounds.extend([(0, None)] * len(self.noise))
        soln = minimize(
            lambda param: -self.log_likelihood(param, inputs, obs),
            init_guess,
            bounds=bounds,
        )

        errs = np.zeros(2)
        yhat = self.model_fn(soln.x[:2], inputs)
        residual = obs - yhat
        var_y = np.var(residual, axis=0)
        for idx in range(2):
            errs[idx] = np.sqrt(var_y[idx] / np.dot(inputs[:, idx], inputs[:, idx].T))
        self._mle = (soln.x, errs)

        if errors:  # standard errrors
            return self._mle
        else:
            return self._mle[0]

    def fit(
        self,
        *args,
        num_walkers=15,
        num_chains=20_000,
        save: bool = False,
        moves: Move = None,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        num_walkers : int, optional
            _description_, by default 15
        num_chains : _type_, optional
            _description_, by default 20_000
        save : bool, optional
            _description_, by default False
        moves : Move, optional
            _description_, by default None
        """
        init_pos = self.sample_prior(num_walkers)
        _, ndim = init_pos.shape
        if save:
            fp = kwargs.get("save_dir", Path(os.getcwd())) / "chain.h5"
            backend = emcee.backends.HDFBackend(str(fp))
            backend.reset(num_walkers, self.n_params)
        else:
            backend = None
        dtype = [("log_prior", float), ("log_ll", float)]
        self.sampler = emcee.EnsembleSampler(
            num_walkers,
            ndim,
            self.log_posterior,
            args=args,
            blobs_dtype=dtype,
            moves=moves,
        )
        start = time()
        self.sampler.run_mcmc(init_pos, num_chains, progress=True)
        if self._samples is not None:
            self._samples = None
        end = time()
        multi_time = end - start
        logger.info("Sampling took {0:.1f} seconds".format(multi_time))

    def get_map(self) -> np.ndarray:
        """
        Return maximum a posteori from chain samples indexed by the arg max of log posterior

        Returns
        -------
        np.ndarrau
            Numpy array of MAP. 
        """
        assert self._samples is not None
        _samples, _log_prob = self._samples.samples, self._samples.log_prob
        return _samples[np.argwhere(_log_prob == _log_prob.max())[0][0]]

    def predict(self, params, inputs):
        """Predict future observations using posterior predictive distribution

        Parameters
        ----------
        params : _type_
            _description_
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        predicted_y = []
        if all(
            [
                isinstance(k, (CustomPrior, stats.distributions.rv_frozen))
                for k in self.noise.values()
            ]
        ):
            noise = params[:, 2:]
        else:
            noise = [list(self.noise.values()) for k in range(len(params))]
        for p, n in zip(params[:, :2], noise):
            rl_model_output: np.ndarray = self.model_fn(p, inputs)
            predicted_y.append(stats.norm.rvs(loc=rl_model_output, scale=np.sqrt(n)))
        predicted_ys = np.stack(predicted_y)
        predicted_means = np.array(
            [np.mean(predicted_ys[:, :, i], axis=0) for i in range(2)]
        )
        predicted_sd = np.array(
            [np.std(predicted_ys[:, :, t], axis=0) for t in range(2)]
        )
        lower_bound = predicted_means - 2 * predicted_sd
        upper_bound = predicted_means + 2 * predicted_sd
        return self.get_map(), lower_bound, upper_bound


class BayesSampler(BaseSampler):
    """
    Base model class for Bayesian parameter inference
    """

    def residual(self, params, inputs, obs):
        y_hat: np.ndarray = self.model_fn(params, inputs)
        return obs - y_hat

    def log_likelihood(self, params, inputs, obs):
        residual = self.residual(params[:2], inputs, obs)
        return np.sum(
            stats.norm.logpdf(
                x=residual,
                loc=0,
                scale=list(self.noise.values())
                if not all(
                    [
                        isinstance(k, (CustomPrior, stats.distributions.rv_frozen))
                        for k in self.noise.values()
                    ]
                )
                else params[2:],
            )
        )

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
        return (
            lp + ll,
            lp,
            ll,
        )


class UncertaintyModel(BaseSampler):
    def likelihood(self, theta, input_x, input_y, var_x, var_y, b=10):
        with np.errstate(divide="ignore", invalid="ignore"):
            sd_x, sd_y = var_x ** 0.5, var_y ** 0.5

            d_1 = theta ** 2 * var_x + var_y
            p_1 = (theta * input_y * var_x + var_y * input_x) / d_1
            p_2 = (var_y * input_x ** 2 + var_x * input_y ** 2) / d_1
            p_3 = (sd_y * sd_x) ** 2 / d_1

            t_1 = p_3 ** 0.5 / (2 * (2 * np.pi) ** 0.5 * sd_y * sd_x)
            t_2 = np.exp(-(p_2 - p_1 ** 2) / (2 * p_3))
            t_3 = special.erf((b - p_1) / (2 * p_3) ** 0.5) - special.erf(
                (-p_1) / (2 * p_3) ** 0.5
            )

            return t_1 * t_2 * t_3

    def log_likelihood(self, params, input_x, input_y, var_x):
        theta, var_y = params
        new_ll = np.log(self.likelihood(theta, input_x, input_y, var_x, var_y))
        return np.sum(new_ll[np.isfinite(new_ll)])

    def joint_likelihood(self, params, input_x, input_y, var_x):
        y_1, y_2 = input_y
        x_1, x_2 = input_x
        G, K, noise_1, noise_2 = params
        var_x1, var_x2 = var_x
        return np.sum(
            [
                self.log_likelihood([G, noise_1], x_1, y_1, var_x1),
                self.log_likelihood([K, noise_2], x_2, y_2, var_x2),
            ]
        )

    def log_posterior(self, params, inputs, obs, var_x):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, -np.inf
        ll = self.joint_likelihood(params, inputs, obs, var_x)
        if not np.isfinite(ll):
            return lp, -np.inf, -np.inf
        return (lp + ll, lp, ll)

