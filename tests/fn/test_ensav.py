"""Test ensemble_average (ensav)."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.ensav import ensav, ensemble_average


class TestEnsembleAverage:
    def test_basic(self):
        segments = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = ensemble_average(segments)
        assert isinstance(result, DescriptiveResult)
        assert np.allclose(result.value, [2.0, 3.0, 4.0])

    def test_single(self):
        s = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(ensemble_average(s).value, [1.0, 2.0, 3.0])

    def test_alias(self):
        assert ensav is ensemble_average
