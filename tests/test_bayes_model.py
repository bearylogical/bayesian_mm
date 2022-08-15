import scipy.stats as stats
import pytest
from src.inference.sampler import BayesSampler


class TestBaseModel:
    def test_fit(self, mock_data):
        priors = dict(
            G=stats.norm(loc=30, scale=10), K=stats.norm(loc=30, scale=10), noise=1
        )
        sampler = BayesSampler(priors)

        x, _, y_noise = mock_data

        sampler.fit(x, y_noise, num_chains=100)

        assert sampler.get_chain() is not None

    def test_predict(self):
        pass
