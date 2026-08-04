"""Tests for rgarb.rangayyan_ar_burg.

Spec: Rangayyan & Krishnan (2024) Sec 7.5 "Autoregressive or All-pole
Modeling" p.369; the Burg-lattice recursion is Sec 8.6.2 p.456. Primary
source: Burg (1975). Burg's method always yields a stable (minimum-phase)
model and reflection coefficients of magnitude < 1 -- both pinned here.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaar import rangayyan_ar_burg


def _ar2(n=4000, a1=0.75, a2=-0.5, seed=31):
    # x[t] = a1 x[t-1] + a2 x[t-2] + e[t]
    e = np.random.default_rng(seed).standard_normal(n + 200)
    x = np.zeros(n + 200)
    for t in range(2, n + 200):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x[200:]


def test_rgarb_recovers_known_ar2_coefficients():
    r = rangayyan_ar_burg(_ar2(), order=2)
    a = np.asarray(r["ar_coeffs"], dtype=float)
    assert a.size == 2
    # sign convention differs between texts, so compare on magnitude-matched
    # prediction form: whichever sign, |a1| ~ 0.75 and |a2| ~ 0.5
    assert abs(abs(a[0]) - 0.75) < 0.06
    assert abs(abs(a[1]) - 0.50) < 0.06


def test_rgarb_reflection_coefficients_are_inside_the_unit_circle():
    # p.456: "The magnitudes of the reflection coefficients are less than
    # unity. The Burg formula always yields a [stable model]."
    k = np.asarray(rangayyan_ar_burg(_ar2(), order=8)["reflection"], dtype=float)
    assert np.all(np.abs(k) < 1.0)


def test_rgarb_model_is_stable():
    # Burg's method always yields a minimum-phase model, so every root of the
    # prediction-error filter A(z) = 1 + a_1 z^-1 + ... lies inside |z| = 1.
    # ar_coeffs ARE the A(z) coefficients, so the polynomial is [1, *a] --
    # negating them tests a different filter and spuriously reports a root at
    # 1.168.
    a = np.asarray(rangayyan_ar_burg(_ar2(), order=6)["ar_coeffs"], dtype=float)
    roots = np.roots(np.concatenate([[1.0], a]))
    assert np.all(np.abs(roots) < 1.0)


def test_rgarb_coefficients_are_the_prediction_error_filter():
    # True process x[t] = 0.75 x[t-1] - 0.5 x[t-2] + e[t]; in A(z) form the
    # coefficients are the negatives, so a ~ [-0.75, +0.50].
    a = np.asarray(rangayyan_ar_burg(_ar2(), order=2)["ar_coeffs"], dtype=float)
    assert a[0] == pytest.approx(-0.75, abs=0.06)
    assert a[1] == pytest.approx(0.50, abs=0.06)


def test_rgarb_innovation_variance_is_positive_and_below_signal_variance():
    x = _ar2()
    r = rangayyan_ar_burg(x, order=2)
    assert r["variance"] > 0.0
    assert r["variance"] < float(np.var(x))


def test_rgarb_rejects_order_not_below_series_length():
    # An AR(p) fit needs more samples than parameters; 5 samples cannot
    # support the default order of 10.
    with pytest.raises(ValueError, match="order"):
        rangayyan_ar_burg(np.arange(5.0))
    with pytest.raises(ValueError, match="order"):
        rangayyan_ar_burg(np.arange(20.0), order=0)
