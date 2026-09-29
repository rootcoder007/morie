"""Tests for morie.fn.prgrc — program recidivism."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import ESRes
from morie.fn.prgrc import program_recidivism


class TestProgramRecidivism:
    def test_lower_rate(self):
        rng = np.random.default_rng(42)
        p = rng.binomial(1, 0.2, 200)
        c = rng.binomial(1, 0.4, 200)
        r = program_recidivism(p, c)
        assert isinstance(r, ESRes)
        assert r.estimate < 0

    def test_too_small(self):
        with pytest.raises(ValueError):
            program_recidivism([1], [0])


def test_risk_difference_and_wald_se():
    import math

    p = [1, 0, 0, 1, 0, 0, 0, 1]
    c = [1, 1, 0, 1, 0, 1, 1, 0, 1]
    rp, rc = 3 / 8, 6 / 9
    se = math.sqrt(rp * (1 - rp) / 8 + rc * (1 - rc) / 9)
    r = program_recidivism(p, c)
    assert r.estimate == pytest.approx(rp - rc, rel=1e-14)
    assert r.se == pytest.approx(se, rel=1e-13)
    assert r.ci_lower == pytest.approx(rp - rc - 1.96 * se, rel=1e-13)
