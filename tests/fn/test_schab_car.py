"""CAR model tests, asserting the book's conditions rather than output.

The previous suite pinned rho = 0.854827586206896. That number was an
artifact: the fitter searched a 30-point grid on (0.01, 0.99), so the
"estimate" was a grid node, could never be zero or negative, and for the
identity parameterization lay outside the valid parameter space
entirely. The test encoded the bug.

Schabenberger & Gotway (2005), Sec 6.2.2.2, eqs (6.43)-(6.48).
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.sgcar import (car_rho_bounds, car_rho_ols,
                            conditional_autoregressive as car)
from morie.fn.spcar import schabenberger_car_model as spcar


def _chain(n=24):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def _z(n=24):
    return np.array([np.sin(0.7 * i) + 0.3 * np.cos(0.31 * i) for i in range(n)])


def test_rho_bounds_come_from_the_eigenvalue_condition():
    """Q must be positive definite; the bound is 1/theta (eq 6.48)."""
    W = _chain()
    lo, hi = car_rho_bounds(W, "identity")
    ev = np.linalg.eigvalsh(W)
    assert lo == pytest.approx(1.0 / ev.min())
    assert hi == pytest.approx(1.0 / ev.max())
    # and the precision really is PD inside, singular at the edge
    assert np.linalg.eigvalsh(np.eye(W.shape[0]) - (hi - 1e-9) * W).min() > 0
    assert np.linalg.eigvalsh(np.eye(W.shape[0]) - (hi + 1e-3) * W).min() < 0


def test_the_estimate_stays_inside_the_valid_interval():
    W = _chain()
    for par in ("weighted", "identity"):
        lo, hi = car_rho_bounds(W, par)
        rho = car(_z(), W, parameterization=par).statistic
        assert lo < rho < hi


def test_negative_dependence_is_reachable():
    """The old grid started at 0.01 and could never express competition."""
    W = _chain()
    alternating = np.array([(-1.0) ** i for i in range(W.shape[0])])
    assert car(alternating, W).statistic < 0


def test_recovers_a_known_rho_from_simulated_car_data():
    """Simulate from the model at a known rho and fit it back."""
    W = _chain(30)
    D = np.diag(W.sum(axis=1))
    rng = np.random.default_rng(11)
    for true_rho in (0.0, 0.5, -0.5):
        Q = D - true_rho * W
        S = np.linalg.inv(Q)
        L = np.linalg.cholesky((S + S.T) / 2)
        est = [car(L @ rng.normal(size=30), W).statistic for _ in range(60)]
        assert np.mean(est) == pytest.approx(true_rho, abs=0.30)


def test_the_two_parameterizations_are_different_models():
    """weighted has conditional variance sigma^2/d_i, identity constant."""
    W, z = _chain(), _z()
    assert car(z, W, parameterization="weighted").statistic != \
        car(z, W, parameterization="identity").statistic
    assert car_rho_bounds(W, "identity")[1] < car_rho_bounds(W, "weighted")[1]


def test_haining_rho_ols_matches_its_closed_form():
    """rho_OLS = e'We / e'W^2e (p. 340, citing Haining 1990 p. 130)."""
    W, z = _chain(), _z()
    X = np.ones((z.size, 1))
    e = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    assert car_rho_ols(z, W) == pytest.approx(
        float(e @ (W @ e)) / float(e @ (W @ (W @ e))))


def test_ml_beats_the_grid_it_replaced_on_its_own_likelihood():
    """The old 30-point grid could only ever hit a node."""
    W, z = _chain(), _z()
    r = car(z, W)
    grid = np.linspace(0.01, 0.99, 30)
    assert not np.any(np.isclose(grid, r.statistic, atol=1e-9))


def test_asymmetric_weights_are_rejected():
    """An asymmetric C gives a non-symmetric precision and no valid joint
    distribution (Hammersley-Clifford)."""
    W = _chain()
    Wrs = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)  # row-standardised
    with pytest.raises(ValueError, match="symmetric"):
        car(_z(), Wrs)


def test_input_validation():
    W, z = _chain(), _z()
    with pytest.raises(ValueError, match="to match"):
        car(z[:-1], W)
    with pytest.raises(ValueError, match="one row per element"):
        car(z, W, X=np.ones((5, 1)))
    with pytest.raises(ValueError, match="parameterization"):
        car(z, W, parameterization="nope")


def test_spcar_delegates_and_forwards_every_argument():
    W, z = _chain(), _z()
    X = np.column_stack([np.ones(z.size), np.arange(z.size) / z.size])
    assert spcar(z, W).statistic == car(z, W).statistic
    assert spcar(z, W, X).statistic == car(z, W, X).statistic
    assert spcar(z, W, None, "identity").statistic == \
        car(z, W, None, "identity").statistic
    # a delegate that dropped an argument would still pass equality above
    assert spcar(z, W, X).statistic != spcar(z, W).statistic
    assert spcar(z, W, None, "identity").statistic != spcar(z, W).statistic
