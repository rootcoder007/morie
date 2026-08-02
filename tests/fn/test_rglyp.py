"""Tests for rglyp.rangayyan_lyapunov.

Spec: Rosenstein, M. T., Collins, J. J. & De Luca, C. J. (1993). A practical
method for calculating largest Lyapunov exponents from small data sets.
Physica D 65(1-2):117-134.

NOT Rangayyan -- the 2024 edition contains no occurrence of "Lyapunov" or
"Rosenstein", so the previous "Ch 7" citation pointed at nothing.

The primary was acquired 2026-07-26, so these tests now assert MAGNITUDE
against the paper's own Table 1 (p.122), hand-transcribed into
tests/fn/fixtures/rglyp.json, at the tolerance the paper states for itself:
"less than +/-10%, and most errors were less than +/-5%".
"""

import json
import pathlib

_FIX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "rglyp.json").read_text()
)
_EXPECT = {s["name"]: s for s in _FIX["systems"]}

from morie.fn import _array_core as np
import pytest

from morie.fn.rglyp import rangayyan_lyapunov


def _henon(n, a=1.4, b=0.3):
    """Henon map, Table 1 row 2: a = 1.4, b = 0.3, dt = 1, lambda_1 = 0.418."""
    x, y = 0.1, 0.3
    out = np.empty(n)
    for i in range(n):
        x, y = 1.0 - a * x * x + y, b * x
        out[i] = x
    return out


def test_matches_rosenstein_table_1_logistic():
    """Table 1, p.122: logistic mu = 4.0, dt = 1, lambda_1 = 0.693."""
    want = _EXPECT["logistic"]["expected_lambda1"]
    got = rangayyan_lyapunov(_logistic(6000), m=2, tau=1, max_t=25, dt=1.0)["lyapunov"]
    assert abs(got - want) / want < 0.10, f"expected {want} +/-10%, got {got}"


def test_matches_rosenstein_table_1_henon():
    """Table 1, p.122: Henon a = 1.4, b = 0.3, dt = 1, lambda_1 = 0.418.

    A second system matters: the logistic map alone could be matched by an
    estimator that happens to recover ln 2 for the wrong reason.
    """
    want = _EXPECT["henon"]["expected_lambda1"]
    got = rangayyan_lyapunov(_henon(6000), m=2, tau=1, max_t=25, dt=1.0)["lyapunov"]
    assert abs(got - want) / want < 0.10, f"expected {want} +/-10%, got {got}"


def test_dt_scales_the_exponent_per_equation_13():
    """Eq (13) divides by dt, so lambda is per unit of dt.

    Halving dt must double the exponent. Without a dt parameter the function
    silently returned a per-SAMPLE rate, which for a sampled flow is wrong by
    a factor of fs while still looking like a plausible number.
    """
    x = _logistic(3000)
    base = rangayyan_lyapunov(x, m=2, tau=1, max_t=25, dt=1.0)["lyapunov"]
    for d in (0.5, 0.1, 2.0):
        got = rangayyan_lyapunov(x, m=2, tau=1, max_t=25, dt=d)["lyapunov"]
        assert np.isclose(got, base / d, rtol=1e-9)


def test_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="`dt` must be positive"):
        rangayyan_lyapunov(_logistic(500), dt=0.0)


def _logistic(n, r=4.0, x0=0.4):
    """Logistic map. At r = 4 the largest Lyapunov exponent is exactly ln 2.

    That is a property of the map -- derived from the invariant density, not
    from any estimator -- which makes it a benchmark independent of the code
    under test.
    """
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def test_identity_chaotic_map_gives_a_positive_exponent():
    """Logistic r=4 is chaotic, so lambda must be clearly positive.

    Kept alongside the Table 1 checks as a cheap sign/order guard that does
    not depend on the fixture.
    """
    lam = rangayyan_lyapunov(_logistic(4000), m=3, tau=1, max_t=30)["lyapunov"]
    assert 0.3 < lam < 0.9, f"expected a clearly positive lambda near ln2, got {lam}"


def test_identity_periodic_signal_gives_a_vanishing_exponent():
    """A periodic orbit does not diverge: neighbouring trajectories stay
    neighbouring, so lambda is 0."""
    per = np.sin(np.linspace(0, 80 * np.pi, 4000))
    lam = rangayyan_lyapunov(per, m=4, tau=8, max_t=30)["lyapunov"]
    assert abs(lam) < 0.05, f"periodic signal should give lambda ~ 0, got {lam}"


def test_identity_chaos_exceeds_periodicity():
    """The ordering is the claim the statistic actually makes."""
    chaotic = rangayyan_lyapunov(_logistic(4000), m=3, tau=1, max_t=30)["lyapunov"]
    periodic = rangayyan_lyapunov(
        np.sin(np.linspace(0, 80 * np.pi, 4000)), m=4, tau=8, max_t=30)["lyapunov"]
    assert chaotic > periodic


def test_identity_scale_invariance():
    """lambda is a rate, so rescaling the signal cannot change it.

    Multiplying x by a scales every separation by |a|, which shifts
    <ln d(t)> by the constant ln|a| and leaves the slope untouched.
    """
    x = _logistic(2000)
    base = rangayyan_lyapunov(x, m=3, tau=1, max_t=25)["lyapunov"]
    for a, b in ((500.0, 0.0), (0.002, 0.0), (1.0, 30.0), (-4.0, -7.0)):
        got = rangayyan_lyapunov(a * x + b, m=3, tau=1, max_t=25)["lyapunov"]
        assert np.isclose(got, base, rtol=1e-9, atol=1e-9)


def test_rejects_a_theiler_window_that_excludes_every_neighbour():
    """If the window swallows all candidates, argmin over an all-inf row
    silently returns index 0 -- a "nearest neighbour" that is nothing of the
    kind, and a lambda computed from it means nothing."""
    x = _logistic(60)
    with pytest.raises(ValueError, match="excludes every neighbour"):
        rangayyan_lyapunov(x, m=3, tau=1, theiler=200)


def test_rejects_array_where_a_scalar_belongs():
    x = _logistic(500)
    with pytest.raises(ValueError, match="must be a scalar integer"):
        rangayyan_lyapunov(x, m=np.arange(5))


def test_rejects_series_too_short_to_embed():
    """Both generated tests fed a series too short to embed and then asserted
    keys the function does not return. The function was right."""
    with pytest.raises(ValueError, match="Series too short"):
        rangayyan_lyapunov(np.array([1.0, 2.0, 3.0]), m=3)


def test_returns_documented_keys():
    res = rangayyan_lyapunov(_logistic(1000), m=3, tau=1, max_t=20)
    for key in ("lyapunov", "divergence_curve", "t"):
        assert key in res
    assert np.isfinite(res["lyapunov"])
