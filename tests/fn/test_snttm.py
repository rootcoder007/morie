"""Tests for morie.fn.snttm — sentence time served."""

import numpy as np
import pytest

from morie.fn._containers import ESRes
from morie.fn.snttm import sentence_time_served


class TestSentenceTimeServed:
    def test_returns_esres(self):
        served = np.array([15, 30, 45])
        sentence = np.array([30, 60, 90])
        result = sentence_time_served(served, sentence)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(0.5)

    def test_full_served(self):
        s = np.array([100, 200])
        result = sentence_time_served(s, s)
        assert result.estimate == pytest.approx(1.0)
