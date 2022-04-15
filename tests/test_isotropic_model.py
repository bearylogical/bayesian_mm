import unittest

import numpy as np

from src.inference.models.model import IsotropicModel


class TestIsotropicModel:
    def test_predict(self):
        m_isotropic = IsotropicModel()
        m_params = (9867.9056, 60776.0301, 1.7777)
        m_inputs = (np.array((0.0575, 0.1086)), 11220.0)
        res = m_isotropic.predict(m_params, m_inputs)
        expected_res = np.array([0.07327565, 0.12437565])
        assert (np.isclose(res, expected_res).all())  # add assertion here


if __name__ == '__main__':
    unittest.main()
