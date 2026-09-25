"""Tests for spthom.schabenberger_thomas_process."""

import math

from morie.fn import _array_core as np

from morie.fn.spthom import schabenberger_thomas_process


def _k_exact(r, rho, sigma):
    """K(r) = pi r^2 + (1 - exp(-r^2 / (4 sigma^2))) / rho."""
    return math.pi * r * r + (1.0 - math.exp(-(r * r) / (4.0 * sigma ** 2))) / rho


def test_spthom_basic():
    """The returned K-function must match the closed form of the docstring."""
    r = [0.0, 0.25, 0.5, 1.0, 2.0]
    rho, mu, sigma = 4.0, 3.0, 0.5
    result = schabenberger_thomas_process(r, rho=rho, mu=mu, sigma=sigma)

    k = [float(v) for v in np.asarray(result["k"]).ravel()]
    k_alias = [float(v) for v in np.asarray(result["k_function"]).ravel()]
    csr = [float(v) for v in np.asarray(result["k_csr"]).ravel()]
    excess = [float(v) for v in np.asarray(result["excess"]).ravel()]

    assert len(k) == 5
    for i, ri in enumerate(r):
        assert abs(k[i] - _k_exact(ri, rho, sigma)) < 1e-12
        assert abs(csr[i] - math.pi * ri * ri) < 1e-12
        assert abs(k[i] - csr[i] - excess[i]) < 1e-12
        assert abs(k_alias[i] - k[i]) < 1e-15
        # Clustering always sits above the Poisson K-function.
        assert excess[i] >= 0.0

    # lambda = rho * mu.
    assert abs(result["lambda"] - 12.0) < 1e-12
    # At r = 0 the excess vanishes and K(0) = 0.
    assert abs(k[0]) < 1e-15
    # The excess climbs towards its 1 / rho ceiling and never reaches it.
    assert excess[1] < excess[2] < excess[3] < excess[4] < 1.0 / rho


def test_spthom_edge():
    """Degenerate parameters are rejected; a scalar r is lifted to length 1."""
    one = schabenberger_thomas_process(0.5, rho=2.0, mu=1.0, sigma=0.25)
    k = [float(v) for v in np.asarray(one["k"]).ravel()]
    assert len(k) == 1
    assert abs(k[0] - _k_exact(0.5, 2.0, 0.25)) < 1e-12

    for bad in (dict(rho=0.0), dict(mu=0.0), dict(sigma=0.0),
                dict(rho=-1.0), dict(sigma=-0.5)):
        try:
            schabenberger_thomas_process([1.0], **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))

    try:
        schabenberger_thomas_process([-1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a negative distance")
