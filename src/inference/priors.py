from typing import Union

import numpy as np


class CustomPrior:
    def pdf(self, param):
        raise NotImplementedError

    def logpdf(self, param: float):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(param))

    def rvs(self, size: Union[tuple, int] = 1):
        return np.random.random(size=size)


class ReferencePrior(CustomPrior):
    """Used for noise

    Parameters
    ----------
    CustomPrior : _type_
        _description_
    """

    def pdf(self, param: float):
        if param < 0:
            return 0.0
        else:
            return 1 / param


class JeffreysPrior(CustomPrior):
    def pdf(self, param: float):
        if param < 0:
            return 0.0
        else:
            return 1.0


class ModelBasedPrior(CustomPrior):
    def pdf(self, param: float):
        if param < 0:
            return 0.0
        else:
            return (1 + param ** 2) ** (-3 / 2)
