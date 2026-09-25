"""Tests for volsv.vol_sv_quasi_lik.

Every expected value is recomputed here by a route the implementation does
not take: a hand-written copy of the documented Kalman recursion, the
closed form the recursion collapses to when sigma_eta -> 0, and the exact
scale equivariance implied by ``log (c r)**2 = log r**2 + 2 log c``.
"""

import math

from morie.fn import _array_core as np

from morie.fn.volsv import vol_sv_quasi_lik

# E[log z**2] = -gamma - log 2 and Var[log z**2] = psi'(1/2) = pi**2 / 2,
# written from their closed forms rather than taken from the module.
_M = -0.5772156649015329 - math.log(2.0)
_V = math.pi * math.pi / 2.0


def _qll(y, mu, phi, sigma):
    """Independent copy of the quasi log-likelihood the docstring defines."""
    a = 0.0
    p = sigma * sigma / (1.0 - phi * phi)
    c = mu + _M
    ll = 0.0
    for yt in y:
        v = yt - c - a
        f = p + _V
        ll -= 0.5 * (math.log(2.0 * math.pi * f) + v * v / f)
        k = p / f
        a += k * v
        p -= k * p
        a *= phi
        p = phi * phi * p + sigma * sigma
    return ll


def _sample(n=40):
    return [float(v) for v in np.random.default_rng(42).normal(0.0, 1.0, n).tolist()]


def test_volsv_matches_hand_kalman_filter():
    """Reported ll equals a hand Kalman filter at the reported parameters."""
    r = _sample()
    res = vol_sv_quasi_lik(r, offset=0.0)

    assert res["n"] == len(r)
    assert res["sweeps"] == 25
    assert res["offset"] == 0.0
    assert res["method"].startswith("SV(1) quasi-likelihood")

    assert -1.0 < res["phi"] < 1.0
    assert res["sigma_eta"] > 0.0

    y = [math.log(v * v) for v in r]
    assert res["ll"] == _qll(y, res["mu"], res["phi"], res["sigma_eta"])

    # The two moment constants are exactly the closed forms above.
    assert _M == -1.2703628454614782
    assert abs(_V - 4.934802200544679) < 1e-12


def test_volsv_iid_returns_collapse_to_the_closed_form():
    """For iid normal returns the fit degenerates to constant log-variance.

    With no persistence in the data the maximiser drives sigma_eta to zero,
    and the recursion then reduces to an iid Gaussian quasi-likelihood in
    ``log r**2`` whose maximiser is ``mean(log r**2) - E[log z**2]``.
    """
    r = _sample()
    y = [math.log(v * v) for v in r]
    res = vol_sv_quasi_lik(r, offset=0.0)

    assert res["sigma_eta"] < 1e-12
    mu_star = sum(y) / len(y) - _M
    assert abs(res["mu"] - mu_star) < 1e-6

    flat_ll = -0.5 * sum(
        math.log(2.0 * math.pi * _V) + (t - mu_star - _M) ** 2 / _V for t in y
    )
    assert abs(res["ll"] - flat_ll) < 1e-9
    # And the reported ll cannot beat that maximum.
    assert res["ll"] <= flat_ll + 1e-9


def test_volsv_is_a_coordinate_optimum():
    """No in-range single-coordinate nudge raises the independent qll."""
    r = _sample()
    res = vol_sv_quasi_lik(r, offset=0.0)
    y = [math.log(v * v) for v in r]
    mu, phi, sig = res["mu"], res["phi"], res["sigma_eta"]
    best = _qll(y, mu, phi, sig)

    for d in (-1e-3, 1e-3):
        assert _qll(y, mu + d, phi, sig) <= best + 1e-12
        if -0.999 <= phi + d <= 0.999:
            assert _qll(y, mu, phi + d, sig) <= best + 1e-12
    # sigma_eta sits at the lower edge of its range, so only growth is in range.
    assert _qll(y, mu, phi, sig + 1e-3) <= best + 1e-12


def test_volsv_scale_equivariance():
    """Scaling the returns by c shifts mu by 2 log c and leaves the rest."""
    r = _sample()
    c = 3.0
    base = vol_sv_quasi_lik(r, offset=0.0)
    scaled = vol_sv_quasi_lik([c * v for v in r], offset=0.0)

    assert abs((scaled["mu"] - base["mu"]) - 2.0 * math.log(c)) < 1e-6
    assert scaled["phi"] == base["phi"]
    # The log 2 pi f terms are unchanged, so the quasi likelihood is too.
    assert abs(scaled["ll"] - base["ll"]) < 1e-9

    # The identity holds exactly for the hand filter, with no optimiser in play.
    y = [math.log(v * v) for v in r]
    ys = [math.log((c * v) ** 2) for v in r]
    assert abs(
        _qll(ys, base["mu"] + 2.0 * math.log(c), 0.7, 0.4) - _qll(y, base["mu"], 0.7, 0.4)
    ) < 1e-9


def test_volsv_edge():
    """Short series are rejected; an explicit init is echoed and improved on."""
    try:
        vol_sv_quasi_lik([0.1] * 9)
    except ValueError:
        pass
    else:
        raise AssertionError("fewer than ten observations must raise")

    for bad in ((0.0, 1.5, 0.3), (0.0, 0.5, 0.0)):
        try:
            vol_sv_quasi_lik(_sample(), init=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range init must raise: %r" % (bad,))

    r = _sample()
    res = vol_sv_quasi_lik(r, init=(0.0, 0.5, 0.3), sweeps=2, offset=1e-8)
    assert res["init"] == [0.0, 0.5, 0.3]
    assert res["sweeps"] == 2

    y = [math.log(v * v + 1e-8) for v in r]
    assert res["ll"] == _qll(y, res["mu"], res["phi"], res["sigma_eta"])
    # Two sweeps already beat the starting point.
    assert res["ll"] > _qll(y, 0.0, 0.5, 0.3)
