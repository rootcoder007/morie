"""Tests for rgapn.rangayyan_approximate_entropy.

Spec: Pincus (1991), PNAS 88(6):2297-2301. NOT Rangayyan -- the 2024 edition
contains no occurrence of "approximate entropy" or "Pincus", so the previous
"Ch 7" citation pointed at nothing.

ApEn's conventions are the mirror image of SampEn's, and the difference is
deliberate: ApEn INCLUDES self-matches, and evaluates phi^m over N-m+1
template vectors against phi^(m+1) over N-m. The asymmetric template count
that is a bug in rgsam is correct here.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgapn import rangayyan_approximate_entropy


def _apen_reference(x, m, r):
    """Literal Pincus (1991), self-matches included, N-m+1 vectors per phi."""
    x = np.asarray(x, float)
    N = x.size
    def phi(mm):
        nT = N - mm + 1
        tot = 0.0
        for i in range(nT):
            c = sum(1 for j in range(nT)
                    if np.max(np.abs(x[i:i+mm] - x[j:j+mm])) <= r)
            tot += np.log(c / nT)          # self-match always counted
        return tot / nT
    return phi(m) - phi(m + 1)


def test_matches_pincus_definition():
    """Implementation == Pincus, re-derived from the definition."""
    rng = np.random.default_rng(20260726)
    for n, m in ((120, 2), (200, 2), (150, 3)):
        x = rng.standard_normal(n)
        r = 0.2 * x.std()
        got = rangayyan_approximate_entropy(x, m=m, r=r)["ApEn"]
        assert np.isclose(got, _apen_reference(x, m, r), rtol=1e-12, atol=1e-12)


def test_self_matches_are_included():
    """Every template matches itself, so no count can be zero.

    This is the property that distinguishes ApEn from SampEn and the reason
    ApEn never diverges: C_i >= 1/nT always, so log() is always finite. If
    self-matches were ever dropped here, a tight tolerance would produce
    -inf instead of a large finite value.
    """
    x = np.random.default_rng(21).standard_normal(80)
    res = rangayyan_approximate_entropy(x, m=2, r=1e-12)
    assert np.isfinite(res["ApEn"]), "ApEn must stay finite -- self-matches included"


def test_identity_constant_signal_has_zero_entropy():
    """A constant series is perfectly regular: every C_i = 1, so phi = 0 and
    ApEn = 0."""
    x = np.full(200, -1.5)
    assert np.isclose(rangayyan_approximate_entropy(x, m=2, r=0.1)["ApEn"], 0.0,
                      atol=1e-12)


def test_identity_periodic_is_more_regular_than_noise():
    t = np.linspace(0, 20 * np.pi, 500)
    sine = np.sin(t)
    noise = np.random.default_rng(23).standard_normal(500)
    assert (rangayyan_approximate_entropy(sine, m=2)["ApEn"]
            < rangayyan_approximate_entropy(noise, m=2)["ApEn"])


def test_identity_scale_invariance_with_relative_tolerance():
    x = np.random.default_rng(29).standard_normal(250)
    base = rangayyan_approximate_entropy(x, m=2)["ApEn"]
    for a, b in ((50.0, 0.0), (0.02, 0.0), (1.0, 12.0), (-4.0, -1.0)):
        assert np.isclose(rangayyan_approximate_entropy(a * x + b, m=2)["ApEn"],
                          base, rtol=1e-9, atol=1e-9)


def test_returns_documented_keys():
    """The generated test asserted an "estimate" key this never promised."""
    res = rangayyan_approximate_entropy(np.random.default_rng(2).standard_normal(200))
    for key in ("ApEn", "phi_m", "phi_m1", "m", "r", "n"):
        assert key in res


def test_rejects_series_shorter_than_m_plus_two():
    """The old edge test passed a single sample and asserted result["n"] == 1,
    expecting a key the function does not return from a computation it cannot
    perform."""
    with pytest.raises(ValueError, match=r"len\(x\) > m \+ 1"):
        rangayyan_approximate_entropy(np.array([42.0]))
