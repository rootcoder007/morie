"""Test complexity (scmpl)."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.scmpl import complexity, scmpl


class TestComplexity:
    def test_sine(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        result = complexity(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_alias(self):
        assert scmpl is complexity
