"""Verification tests for information_theory_mackay3e9.bcoinpri.

The expected values are recomputed from MacKay (2003) eq. (3.9) p. 51 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay3e9 import bcoinpri


def test_bcoinpri_is_the_uniform_density_on_the_unit_interval():
    for pa in (0.0, 0.25, 0.5, 1.0):
        res = bcoinpri(pa)
        assert res["density"] == pytest.approx(1.0, abs=1e-12)
        assert res["logdensity"] == pytest.approx(0.0, abs=1e-12)
    outside = bcoinpri(1.5)
    assert outside["density"] == 0.0
    assert outside["logdensity"] == -math.inf


def test_bcoinpri_integrates_to_one_over_the_unit_interval():
    n = 1000
    total = sum(bcoinpri(i / n)["density"] for i in range(n)) / n
    assert total == pytest.approx(1.0, abs=1e-9)
