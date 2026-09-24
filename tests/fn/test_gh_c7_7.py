"""Verification tests for gh_c7_7.

Ghosal and van der Vaart (2017), sec. 7.3.3, Whittle spectral-density estimation.
"""

import math

import pytest

from morie.fn.gh_c7_7 import ghosal_spec_dens_con


def test_whittle_estimation_returns_a_non_negative_spectral_density():
    # Sec. 7.3.3: the periodogram ordinates are asymptotically
    # independent exponentials with mean f(omega_j), so both the
    # periodogram and the fitted density are non-negative
    x = [0.5, -0.2, 0.9, -0.7, 0.3, 0.1, -0.4, 0.8, 0.2, -0.6]
    res = ghosal_spec_dens_con(x)
    dens = [float(v) for v in res["spectral_density"]]
    per = [float(v) for v in res["periodogram"]]
    assert dens and all(v >= 0.0 for v in dens)
    assert all(v >= 0.0 for v in per)
    assert len(dens) == len(per) == len(list(res["freqs"]))


def test_the_frequencies_are_the_fourier_grid():
    x = [0.5, -0.2, 0.9, -0.7, 0.3, 0.1, -0.4, 0.8, 0.2, -0.6]
    res = ghosal_spec_dens_con(x)
    n = res["n"]
    freqs = [float(v) for v in res["freqs"]]
    for j, w in enumerate(freqs, start=1):
        assert w == pytest.approx(2.0 * math.pi * j / n, rel=1e-12)


def test_the_whittle_log_likelihood_is_finite():
    x = [0.5, -0.2, 0.9, -0.7, 0.3, 0.1, -0.4, 0.8, 0.2, -0.6]
    assert math.isfinite(float(ghosal_spec_dens_con(x)["whittle_loglik"]))


def test_a_series_shorter_than_the_minimum_is_refused():
    with pytest.raises(ValueError):
        ghosal_spec_dens_con([0.1, 0.2, 0.3])
