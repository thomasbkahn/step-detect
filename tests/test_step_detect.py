"""
To run (from root directory):
python -m unittest discover tests
"""
import unittest
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import step_detect


def step_detect_test_data(amplitude, sd):
    """
    Function to generate time series data with defined step changes (from 0 to ``A``) and noise with standard deviation
    of ``sd`` (using fixed random seed so that code will be consistent in either failing or passing the unit test
    functions).
    :param float amplitude: Amplitude of step changes
    :param float sd: Standard deviation of the normal distribution that noise is drawn from to add noise
    to data (deterministic since using fixed random seed)
    :return: tuple of x and y data
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    x = np.linspace(0, 500, 1000)
    y = np.zeros_like(x)
    y[np.logical_and(x > 50, x < 150)] = amplitude
    y[np.logical_and(x > 300, x < 350)] = amplitude
    y[np.logical_and(x > 357, x < 437.5)] = amplitude
    prng = np.random.RandomState(0)
    y += prng.normal(scale=sd, size=x.shape)
    return y


class TestStepDetectFunctions(unittest.TestCase):
    def setUp(self):
        """
        Method that initializes data before running the tests.
        In this case we generate signals with known analytical properties
        such that it's easier to identify the correctness of the
        functions.
        """
        self.jump_magnitude = 1000
        self.series1 = step_detect_test_data(self.jump_magnitude, self.jump_magnitude / 10)
        self.series1_filtered = gaussian_filter1d(self.series1.flatten(), 2, order=1)
        self.series2 = np.array([1704.70739332, 1687.10112437, 1681.68248344, 1678.11859602,
                                 1683.25987459, 1676.34517265, 1683.75263377, 1670.12514163,
                                 1694.67019815, 1675.8585799, 1698.97665797, 1681.66520782,
                                 1700.68003068, 1671.82624614, 1666.37968373, 3211.59480037,
                                 3303.86229941, 3289.33136964, 3299.01543621, 3286.30224878,
                                 3292.78710246, 3297.47755896, 2574.27847639, 1665.98144833,
                                 1682.55691805, 1681.6211562, 1671.7071523, 1670.22268554,
                                 1656.43500242, 1661.8255372])
        self.series2_filtered = gaussian_filter1d(self.series2.flatten(), 2, order=1)
        self.series3 = np.array([1883.87443401, 1888.4688974, 1893.1127929, 1892.13823514,
                                 1891.87245162, 1882.15734314, 1909.32780929, 1890.95291424,
                                 1891.69441096, 1895.73540552, 1879.59040827, 2207.4511012,
                                 3594.28900568, 3539.73771894, 3512.77316641, 3474.76579655,
                                 3445.91259433, 3449.58999, 3538.07678893, 5071.54845559,
                                 5023.52196979, 5008.00727191, 4997.53409712, 4973.39290886,
                                 4944.09166901, 4943.28154164])
        self.series3_filtered = gaussian_filter1d(self.series3.flatten(), 2, order=1)

    def test_find_steps(self):
        steps = step_detect.find_steps(self.series1_filtered, 50)
        self.assertTrue(len(steps) == 6, "length of jumps for test series1 ({}) != 6".format(len(steps)))
        for i, true_ix in enumerate([100, 300, 600, 700, 714, 875]):
            self.assertTrue(np.abs(steps[i] - true_ix) < 3,
                            "jump location {} more than 2 off from truth for test series 1".format(steps[i]))
        steps = step_detect.find_steps(self.series2_filtered, 10)
        self.assertTrue(len(steps) == 2, "length of jumps for test series2 ({} != 2)".format(len(steps)))
