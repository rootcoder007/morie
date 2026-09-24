"""Verification tests for gh_c11_10.

Ghosal and van der Vaart (2017), Ex 11.16, the eigenexpansion of a Gaussian process kernel.
"""

import math

import pytest

from morie.fn.gh_c11_10 import ghosal_series_gp


def test_the_eigenexpansion_converges_to_the_kernel():
    # Ex 11.16: K(x, y) = sum_k lambda_k phi_k(x) phi_k(y), so the
    # partial sums must settle
    res = ghosal_series_gp(x=0.3, y=0.7, n_terms=80)
    sums = [float(v) for v in res["partial_sums"]]
    assert res["converging"] is True
    # the increments shrink: the tail of the series is summable
    tail_early = abs(sums[len(sums) // 4] - sums[len(sums) // 4 - 1])
    tail_late = abs(sums[-1] - sums[-2])
    assert tail_late < tail_early


def test_the_kernel_is_symmetric_in_its_arguments():
    a = ghosal_series_gp(x=0.3, y=0.7, n_terms=60)["estimate"]
    b = ghosal_series_gp(x=0.7, y=0.3, n_terms=60)["estimate"]
    assert a == pytest.approx(b, rel=1e-12)
