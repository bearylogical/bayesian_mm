from scipy.stats import entropy
import numpy as np


def kl_divergence(p1: np.ndarray, p2: np.ndarray):
    """

    Parameters
    ----------
    p1 : proposal distribution (prior)
    p2 : target distribution (posterior)

    Returns
    -------

    """
    return entropy(p1, p2, base=2)
