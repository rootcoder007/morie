"""Verification tests for km074.

Kamath, Keenan, Somers and Sorenson (2024), ch 5, the DPO preference with the partition function substituted, eq. 5.10. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km074 import kamath_ch5_dpo_pref_substituted


def test_the_substituted_preference_drops_the_partition_function():
    # p* = sigma(beta log(pi*(y1)/pi_ref(y1)) - beta log(pi*(y2)/pi_ref(y2)))
    res = kamath_ch5_dpo_pref_substituted([0.75, 0.25], [0.5, 0.5], 1.0, Z=1000.0)
    logit = math.log(0.75 / 0.5) - math.log(0.25 / 0.5)
    assert res["estimate"] == pytest.approx(1.0 / (1.0 + math.exp(-logit)),
                                            rel=1e-12)
    assert round(res["estimate"], 12) == pytest.approx(0.75, abs=1e-12)
    assert res["z_terms_cancel"] is True


def test_the_partition_function_cannot_change_the_preference():
    a = kamath_ch5_dpo_pref_substituted([0.75, 0.25], [0.5, 0.5], 1.0, Z=2.0)["estimate"]
    b = kamath_ch5_dpo_pref_substituted([0.75, 0.25], [0.5, 0.5], 1.0, Z=1e6)["estimate"]
    assert a == pytest.approx(b, rel=1e-12)


def test_a_larger_coefficient_sharpens_the_preference():
    res = kamath_ch5_dpo_pref_substituted([0.75, 0.25], [0.5, 0.5], 2.0, Z=1.0)
    # sigma(2 log 3) = 9 / 10
    assert res["estimate"] == pytest.approx(0.9, rel=1e-12)
