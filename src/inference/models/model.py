import numpy as np
import scipy.stats
from scipy.optimize import minimize


class BayesModel:
    """
    Base model class for Bayesian parameter inference
    """

    def __init__(self,
                 noise=1,
                 length_scale: float = 100,
                 **kwargs):
        self.noise = noise
        self.params = kwargs
        self.param_dists = list(self.params.values())
        self.param_names = list(self.params.keys())
        self.n_params = len(self.params)
        self.length_scale = length_scale

    def __str__(self):
        return self.params

    def sample_prior(self, size):
        return np.array([dist.rvs(size) for dist in self.param_dists]).T

    def log_prior(self, params):
        return sum(dist.logpdf(param) for param, dist in zip(params, self.param_dists))

    def log_likelihood(self, params, inputs, obs):
        rl_model = self.predict(params, inputs)
        return scipy.stats.multivariate_normal.logpdf(rl_model.T.flatten() * self.length_scale,
                                                      obs.T.flatten() * self.length_scale,
                                                      self.noise)

    def negative_log_likehood(self, *args):
        return -self.log_likelihood(*args)

    def log_posterior(self, params, inputs, obs):
        return self.log_prior(params) + self.log_likelihood(params, inputs, obs)

    def max_likelihood(self, inputs, obs):
        init_guss = self.sample_prior(1)
        soln = minimize(self.negative_log_likehood, init_guss, args=(inputs, obs))
        return soln.x

    def predict(self, params, inputs) -> np.ndarray:
        pass

    def save(self, save_posterior: bool = False):
        """

        Parameters
        ----------
        save_posterior: flag to store the posterior to be used as the next prior.

        Returns
        -------

        """
        pass

    def load_prior(self):
        pass


class IsotropicModel(BayesModel):
    def predict(self, params, inputs) -> np.ndarray:
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
        rl_0, p_0 = inputs[:, :2], inputs[:, 2]

        A = np.array([[1, -1],
                      [2, 1]])
        A_inv = np.linalg.inv(A)

        p_wall = p_factor * p_0
        delta_P = p_wall - p_0
        p_avg = 1 / 3 * (2 * p_wall + p_0)
        b1 = delta_P / (2 * G)
        b2 = p_avg / (2 * K)
        b = np.array([b1, b2])
        eps = A_inv.dot(b)  # [er, ez]

        return rl_0 - eps.T * rl_0
