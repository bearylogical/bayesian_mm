import unittest
import numpy as np
import src.utils.transforms as transforms

class TestTransforms(unittest.TestCase):
    def test_normalise_bands(self):
        m_rl_band = np.array([117.9, 313.0])
        img_size = (2880, 2048)
        res = transforms.normalise_bands(m_rl_band, img_size=img_size)
        exp_res = (0.057568359375, 0.10868055555555556)
        self.assertTrue(np.isclose(res, exp_res).all())


if __name__ == '__main__':
    unittest.main()
