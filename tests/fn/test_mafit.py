"""Test mafit."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mafit import mafit


def test_mafit_basic():
    """MA(1) fitting."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(100)
    # MA(1) with theta = -0.5
    y = np.zeros(100)
    y[0] = eps[0]
    for t in range(1, 100):
        y[t] = eps[t] - 0.5 * eps[t - 1]

    r = mafit(y, q=1)
    assert isinstance(r.ma_coeff, np.ndarray)
    assert r.ma_coeff.shape == (1,)
    assert r.sigma2 > 0


def test_mafit_higher_order():
    """MA(2) model."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(150)
    y = np.zeros(150)
    y[0] = eps[0]
    y[1] = eps[1]
    for t in range(2, 150):
        y[t] = eps[t] - 0.3 * eps[t - 1] + 0.2 * eps[t - 2]

    r = mafit(y, q=2)
    assert r.ma_coeff.shape == (2,)
    assert r.acf.shape == (3,)  # q+1


def test_mafit_acf_properties():
    """ACF properties."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    r = mafit(y, q=2)

    assert r.acf[0] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.abs(r.acf) <= 1.0)


def test_mafit_validation():
    """Input validation."""
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        mafit(y, q=3)  # q >= n


def test_mafit_minimises_the_conditional_sum_of_squares():
    """MA(1), X_t = Z_t - 0.5 Z_{t-1}: the fitted theta is the minimiser
    of sum_t eps_t^2 with eps_t = x_t - theta eps_{t-1} (eps_0 = x_0),
    located here on a 0.001 grid over the invertible range; the fit
    must sit within one grid step of it, and on the right side of 0."""
    rng = np.random.default_rng(42)
    e = rng.standard_normal(200).tolist()
    x = [e[0]] + [e[t] - 0.5 * e[t - 1] for t in range(1, 200)]
    m = sum(x) / len(x)
    x = [v - m for v in x]

    def css(th):
        eps, s = x[0], 0.0
        for t in range(1, len(x)):
            eps = x[t] - th * eps
            s += eps * eps
        return s

    grid = [i / 1000 for i in range(-999, 1000)]
    best = min(grid, key=css)
    r = mafit(np.array(x), q=1)
    th = float(r.ma_coeff[0])
    assert abs(th - best) <= 0.001
    assert th < 0


def test_mafit_acf_is_the_brockwell_davis_ma_acf():
    """rho(k) = (theta_k + sum_j theta_j theta_{j+k}) / (1 + sum theta^2)
    for the fitted MA(3) coefficients (statsmodels' arma_acf agrees)."""
    y = np.random.default_rng(3).standard_normal(150)
    r = mafit(y, q=3)
    th = [float(v) for v in r.ma_coeff]
    den = 1 + sum(v * v for v in th)
    exp = [1.0] + [(th[k - 1] + sum(th[j - 1] * th[j + k - 1] for j in range(1, 4 - k))) / den
                   for k in range(1, 4)]
    assert [float(v) for v in r.acf] == pytest.approx(exp, rel=1e-14, abs=1e-15)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest
import importlib as _importlib

_doctest_module = _importlib.import_module("morie.fn.mafit")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
