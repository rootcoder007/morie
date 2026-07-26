"""Tests for rgdfa.rangayyan_dfa.

Spec: Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides.
Physical Review E 49(2):1685-1689, method on p.1685.

NOT Rangayyan. The 2024 edition mentions "detrended fluctuation" four times,
every one a citation to someone else's application rather than a treatment, so
the previous "Ch 7" citation pointed at nothing.

Peng gives no transcribable input/output worked example -- the paper's anchors
are the scaling exponents themselves (Fig. 3: alpha = 0.51 for the uncorrelated
control, 0.61 for the correlated one). Those anchors are what the identity
tests below pin, alongside a re-derivation of F_d(l) from the definition.
"""

import numpy as np
import pytest

from morie.fn.rgdfa import rangayyan_dfa


def _F_reference(x, n, order=1):
    """F_d(l) straight from p.1685, written from the paper not the code.

    "Calculate the variance about the detrended walk for each box, and
    calculate the average of these variances over all the boxes of size l."
    So: mean the per-box residual variances, then take the square root.
    """
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    nseg = x.size // n
    v = []
    for k in range(nseg):
        seg = y[k * n : (k + 1) * n]
        t = np.arange(n)
        v.append(np.mean((seg - np.polyval(np.polyfit(t, seg, order), t)) ** 2))
    return np.sqrt(np.mean(v))


def test_fluctuation_matches_the_definition():
    """F(n) as reported == F_d(n) re-derived from Peng's step (2)."""
    x = np.random.default_rng(51).standard_normal(600)
    res = rangayyan_dfa(x)
    for n, F in zip(res["scales"], res["F"]):
        assert np.isclose(F, _F_reference(x, int(n)), rtol=1e-12)


def test_it_is_the_root_mean_variance_not_the_mean_root():
    """sqrt(mean(var)) and mean(sqrt(var)) are different quantities.

    Peng averages the variances and then takes the root. Averaging the
    per-box standard deviations instead gives a systematically smaller F by
    Jensen's inequality -- a plausible-looking but wrong implementation that
    no smoke test would catch.
    """
    x = np.random.default_rng(53).standard_normal(400)
    y = np.cumsum(x - x.mean())
    n = 32
    nseg = x.size // n
    v = []
    for k in range(nseg):
        seg = y[k * n : (k + 1) * n]
        t = np.arange(n)
        v.append(np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2))
    assert np.sqrt(np.mean(v)) > np.mean(np.sqrt(v))          # Jensen
    # Two scales, because a slope needs two points; only the n-box F is
    # compared here.
    res = rangayyan_dfa(x, scales=[n, 2 * n])
    assert np.isclose(res["F"][0], np.sqrt(np.mean(v)), rtol=1e-12)


def test_identity_white_noise_gives_alpha_one_half():
    """Peng p.1685: with no long-range correlation F_d(l) ~ l^(1/2).

    Their own uncorrelated control measured 0.51 (Fig. 3).
    """
    x = np.random.default_rng(57).standard_normal(8000)
    alpha = rangayyan_dfa(x)["alpha"]
    assert 0.45 < alpha < 0.55, f"white noise should give alpha ~ 0.5, got {alpha}"


def test_identity_random_walk_gives_alpha_three_halves():
    """Integrating white noise adds exactly 1 to the exponent.

    A Brownian series is the cumulative sum of white noise, so alpha goes
    0.5 -> 1.5. This is the sharpest available check that the integration
    step and the scaling fit are wired together correctly.
    """
    w = np.cumsum(np.random.default_rng(59).standard_normal(8000))
    alpha = rangayyan_dfa(w)["alpha"]
    assert 1.4 < alpha < 1.6, f"random walk should give alpha ~ 1.5, got {alpha}"


def test_identity_scale_invariance():
    """alpha is a scaling exponent: rescaling x cannot move it.

    Multiplying x by a scales every F(n) by |a|, shifting log F by a constant
    and leaving the slope untouched.
    """
    x = np.random.default_rng(61).standard_normal(2000)
    base = rangayyan_dfa(x)["alpha"]
    for a, b in ((250.0, 0.0), (0.004, 0.0), (1.0, 60.0), (-5.0, -3.0)):
        assert np.isclose(rangayyan_dfa(a * x + b)["alpha"], base, rtol=1e-9)


def test_rejects_boxes_too_small_for_the_detrending_order():
    """A box of order+1 points fits the polynomial exactly, so every residual
    is zero, F(n) = 0 and log F = -inf silently poisons the slope. numpy would
    only raise a RankWarning."""
    x = np.random.default_rng(63).standard_normal(500)
    with pytest.raises(ValueError, match="too small for order"):
        rangayyan_dfa(x, scales=[3, 8, 16], order=2)


def test_rejects_series_shorter_than_the_documented_minimum():
    """Both generated tests passed 5 and 1 samples against a documented
    32-sample floor, then asserted keys ("estimate", "n") the function does
    not return. The function was right; the tests were wrong."""
    with pytest.raises(ValueError, match="at least 32 samples"):
        rangayyan_dfa(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="at least 32 samples"):
        rangayyan_dfa(np.array([42.0]))


def test_rejects_a_single_scale():
    """One box size cannot determine a slope.

    polyfit would fit an arbitrary line through the single point and only
    warn, returning an alpha that looks like a number and means nothing.
    """
    x = np.random.default_rng(65).standard_normal(500)
    with pytest.raises(ValueError, match="at least 2 usable box sizes"):
        rangayyan_dfa(x, scales=[32])


def test_returns_documented_keys():
    res = rangayyan_dfa(np.random.default_rng(67).standard_normal(500))
    for key in ("alpha", "scales", "F", "log_scales", "log_F"):
        assert key in res
    assert np.isfinite(res["alpha"])
