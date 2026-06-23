"""Tests for morie.fn.prgsv — program survival."""

import numpy as np
import pytest

from morie.fn._containers import DescriptiveResult
from morie.fn.prgsv import program_survival


class TestProgramSurvival:
    def test_basic(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(12, 100)
        completed = rng.binomial(1, 0.7, 100)
        r = program_survival(times, completed)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n"] == 100

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            program_survival([], [])
