# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity tests for the C++ Hawkes likelihood kernel (Phase 2).

The compiled hawkes_ll_exp_const is checked against the reference
morie/tps_hawkes_jit.py::_ll_exp_const (same arithmetic). Requires the
built extension.
"""

from morie.fn import _array_core as np
import pytest

core = pytest.importorskip("morie._core")

from morie.tps_hawkes_jit import _ll_exp_const


def _event_times(n, rate, seed):
    """Sorted event times from an exponential inter-arrival process."""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def _buf(t):
    """A float64 buffer the compiled kernel can take on ANY Python.

    morie's marr exposes a buffer through __buffer__, which is PEP 688
    and therefore Python 3.12+ only; on 3.10 nanobind cannot see it and
    rejects the call. array.array carries the buffer protocol on every
    supported version, and is what tps_hawkes_jit hands the kernel in
    production. The reference implementations keep taking the native
    array, so only the kernel calls are wrapped.
    """
    import array as _pyarray

    return _pyarray.array("d", [float(v) for v in t])


@pytest.mark.parametrize(
    "a0,eta,beta",
    [
        (-1.0, 0.30, 1.5),
        (0.5, 0.60, 0.8),
        (-2.0, 0.10, 5.0),
        (1.0, 0.85, 0.2),
    ],
)
def test_hawkes_ll_exp_const_parity(a0, eta, beta):
    t = _event_times(400, rate=2.0, seed=11)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_exp_const(_buf(t), T, a0, eta, beta)
    ref = float(_ll_exp_const(t, T, a0, eta, beta))
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-9)


def test_hawkes_ll_exp_const_infeasible_returns_sentinel():
    t = _event_times(50, rate=1.0, seed=3)
    T = float(t[-1]) + 1.0
    # eta outside (1e-6, 0.999)
    assert core.hawkes_ll_exp_const(_buf(t), T, 0.0, 1.5, 1.0) == 1e12
    # beta outside (0.05, 30.0)
    assert core.hawkes_ll_exp_const(_buf(t), T, 0.0, 0.3, 100.0) == 1e12
    # a0 outside (-20, 20)
    assert core.hawkes_ll_exp_const(_buf(t), T, 50.0, 0.3, 1.0) == 1e12


def test_neg_loglik_jit_routes_through_core():
    # neg_loglik_jit's exp/constant path uses the C++ core
    from morie.tps_hawkes_jit import has_jit_path, neg_loglik_jit

    assert has_jit_path("exponential", "constant") is True
    t = _event_times(200, rate=1.5, seed=8)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 1.2])
    got = neg_loglik_jit(theta, t, T, "exponential", "constant")
    ref = float(_ll_exp_const(t, T, -1.0, 0.4, 1.2))
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(
    "a0,eta,alpha,lam",
    [
        (-1.0, 0.30, 1.5, 2.0),
        (0.5, 0.60, 0.8, 5.0),
        (-2.0, 0.10, 3.0, 1.0),
    ],
)
def test_hawkes_ll_weibull_const_parity(a0, eta, alpha, lam):
    from morie.tps_hawkes_jit import _ll_weibull_const

    t = _event_times(250, rate=2.0, seed=13)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_weibull_const(_buf(t), T, a0, eta, alpha, lam)
    ref = float(_ll_weibull_const(t, T, a0, eta, alpha, lam))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_hawkes_ll_weibull_const_infeasible_returns_sentinel():
    t = _event_times(40, rate=1.0, seed=5)
    T = float(t[-1]) + 1.0
    assert core.hawkes_ll_weibull_const(_buf(t), T, 0.0, 1.5, 1.5, 2.0) == 1e12
    assert core.hawkes_ll_weibull_const(_buf(t), T, 0.0, 0.3, 50.0, 2.0) == 1e12


def test_neg_loglik_jit_routes_weibull_through_core():
    from morie.tps_hawkes_jit import _ll_weibull_const, has_jit_path, neg_loglik_jit

    assert has_jit_path("weibull", "constant") is True
    t = _event_times(180, rate=1.5, seed=9)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 1.5, 2.0])
    got = neg_loglik_jit(theta, t, T, "weibull", "constant")
    ref = float(_ll_weibull_const(t, T, -1.0, 0.4, 1.5, 2.0))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    "a0,eta,alpha,lam",
    [
        (-1.0, 0.30, 1.5, 2.0),
        (0.5, 0.60, 0.8, 5.0),
        (-2.0, 0.10, 3.0, 1.0),
        (0.0, 0.40, 5.0, 0.5),
    ],
)
def test_hawkes_weibull_trunc_is_exact(a0, eta, alpha, lam):
    # the O(n*w) sliding-window form is BIT-IDENTICAL to the O(n^2)
    # version -- the truncated terms underflow to exactly zero.
    t = _event_times(400, rate=2.0, seed=13)
    T = float(t[-1]) + 1.0
    exact = core.hawkes_ll_weibull_const(_buf(t), T, a0, eta, alpha, lam)
    trunc = core.hawkes_ll_weibull_const_trunc(_buf(t), T, a0, eta, alpha, lam)
    assert trunc == exact


@pytest.mark.parametrize(
    "a0,eta,alpha,beta",
    [
        (-1.0, 0.30, 1.5, 2.0),
        (0.5, 0.60, 0.8, 5.0),
        (-2.0, 0.10, 3.0, 1.0),
        (0.0, 0.40, 5.0, 4.0),
        (-1.0, 0.40, 1.0, 3.0),
    ],
)
def test_hawkes_gamma_trunc_is_exact(a0, eta, alpha, beta):
    # the O(n*w) sliding-window form is BIT-IDENTICAL to the O(n^2)
    # gamma version -- the truncated terms underflow to exactly zero.
    t = _event_times(400, rate=2.0, seed=13)
    T = float(t[-1]) + 1.0
    exact = core.hawkes_ll_gamma_const(_buf(t), T, a0, eta, alpha, beta)
    trunc = core.hawkes_ll_gamma_const_trunc(_buf(t), T, a0, eta, alpha, beta)
    assert trunc == exact


@pytest.mark.parametrize(
    "a0,eta,alpha,c",
    [
        (-1.0, 0.30, 2.0, 1.0),
        (0.5, 0.60, 3.5, 0.5),
        (-2.0, 0.10, 1.5, 2.0),
    ],
)
def test_hawkes_ll_lomax_const_parity(a0, eta, alpha, c):
    from morie.tps_hawkes_jit import _ll_lomax_const

    t = _event_times(250, rate=2.0, seed=17)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_lomax_const(_buf(t), T, a0, eta, alpha, c)
    ref = float(_ll_lomax_const(t, T, a0, eta, alpha, c))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_lomax_through_core():
    from morie.tps_hawkes_jit import _ll_lomax_const, has_jit_path, neg_loglik_jit

    assert has_jit_path("lomax", "constant") is True
    t = _event_times(180, rate=1.5, seed=21)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 2.0, 1.0])
    got = neg_loglik_jit(theta, t, T, "lomax", "constant")
    ref = float(_ll_lomax_const(t, T, -1.0, 0.4, 2.0, 1.0))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)
    # alpha <= 1.001 -> the dispatch returns the infeasible sentinel
    bad = neg_loglik_jit(np.array([0.0, 0.3, 1.0, 1.0]), t, T, "lomax", "constant")
    assert bad == 1e12


@pytest.mark.parametrize(
    "a0,eta,alpha,beta",
    [
        (-1.0, 0.30, 2.0, 1.5),
        (0.5, 0.60, 0.8, 3.0),
        (-2.0, 0.10, 5.0, 0.5),
    ],
)
def test_hawkes_ll_gamma_const_parity(a0, eta, alpha, beta):
    from morie.tps_hawkes_jit import _ll_gamma_const

    t = _event_times(250, rate=2.0, seed=23)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_gamma_const(_buf(t), T, a0, eta, alpha, beta)
    ref = float(_ll_gamma_const(t, T, a0, eta, alpha, beta))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_gamma_through_core():
    from morie.tps_hawkes_jit import _ll_gamma_const, has_jit_path, neg_loglik_jit

    assert has_jit_path("gamma", "constant") is True
    t = _event_times(180, rate=1.5, seed=27)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 2.0, 1.5])
    got = neg_loglik_jit(theta, t, T, "gamma", "constant")
    ref = float(_ll_gamma_const(t, T, -1.0, 0.4, 2.0, 1.5))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)
    # alpha outside (0.05, 20) -> infeasible sentinel
    assert core.hawkes_ll_gamma_const(_buf(t), T, 0.0, 0.3, 50.0, 1.5) == 1e12


@pytest.mark.parametrize(
    "a0,a1,a2,a3,eta,beta",
    [
        (-1.0, 0.2, 0.3, -0.1, 0.30, 1.5),
        (0.0, -0.5, 0.1, 0.4, 0.55, 0.8),
    ],
)
def test_hawkes_ll_exp_sin_parity(a0, a1, a2, a3, eta, beta):
    from morie.tps_hawkes_jit import _ll_exp_sin, _sin_grid

    t = _event_times(250, rate=2.0, seed=31)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_exp_sin(_buf(t), T, a0, a1, a2, a3, eta, beta, grid, grid_vals)
    ref = float(_ll_exp_sin(t, T, a0, a1, a2, a3, eta, beta, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_exp_sin_through_core():
    from morie.tps_hawkes_jit import _ll_exp_sin, _sin_grid, has_jit_path, neg_loglik_jit

    assert has_jit_path("exponential", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=33)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 1.2])
    got = neg_loglik_jit(theta, t, T, "exponential", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_exp_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 1.2, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    "a0,a1,a2,a3,eta,alpha,lam",
    [
        (-1.0, 0.2, 0.3, -0.1, 0.30, 1.5, 2.0),
        (0.0, -0.5, 0.1, 0.4, 0.55, 0.8, 5.0),
    ],
)
def test_hawkes_ll_weibull_sin_parity(a0, a1, a2, a3, eta, alpha, lam):
    from morie.tps_hawkes_jit import _ll_weibull_sin, _sin_grid

    t = _event_times(250, rate=2.0, seed=37)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_weibull_sin(_buf(t), T, a0, a1, a2, a3, eta, alpha, lam, grid, grid_vals)
    ref = float(_ll_weibull_sin(t, T, a0, a1, a2, a3, eta, alpha, lam, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_weibull_sin_through_core():
    from morie.tps_hawkes_jit import _ll_weibull_sin, _sin_grid, has_jit_path, neg_loglik_jit

    assert has_jit_path("weibull", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=39)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 1.5, 2.0])
    got = neg_loglik_jit(theta, t, T, "weibull", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_weibull_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 1.5, 2.0, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    "a0,a1,a2,a3,eta,alpha,c",
    [
        (-1.0, 0.2, 0.3, -0.1, 0.30, 2.0, 1.0),
        (0.0, -0.5, 0.1, 0.4, 0.55, 3.5, 0.5),
    ],
)
def test_hawkes_ll_lomax_sin_parity(a0, a1, a2, a3, eta, alpha, c):
    from morie.tps_hawkes_jit import _ll_lomax_sin, _sin_grid

    t = _event_times(250, rate=2.0, seed=41)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_lomax_sin(_buf(t), T, a0, a1, a2, a3, eta, alpha, c, grid, grid_vals)
    ref = float(_ll_lomax_sin(t, T, a0, a1, a2, a3, eta, alpha, c, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_lomax_sin_through_core():
    from morie.tps_hawkes_jit import _ll_lomax_sin, _sin_grid, has_jit_path, neg_loglik_jit

    assert has_jit_path("lomax", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=43)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 2.0, 1.0])
    got = neg_loglik_jit(theta, t, T, "lomax", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_lomax_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 2.0, 1.0, grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


# --- sum-of-exponentials (SoE) engine, task #73 -----------------------


def _soe_ll_reference(t, T, nu, eta, w, beta):
    """Brute-force O(n^2) reference for the SoE Hawkes likelihood:
    g(u) = sum_m w[m]*exp(-beta[m]*u)."""
    import math

    w = np.asarray(w, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = len(t)
    log_sum = 0.0
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += float(np.sum(w * np.exp(-beta * (t[i] - t[j]))))
        log_sum += math.log(nu + eta * s)
    integral = nu * T
    for i in range(n):
        u = T - t[i]
        integral += eta * float(np.sum((w / beta) * (1.0 - np.exp(-beta * u))))
    return -(log_sum - integral)


def test_hawkes_ll_soe_reduces_to_exponential():
    # SoE with M=1, w=[beta], beta=[beta] is exactly the exp kernel
    import math

    t = _event_times(300, rate=2.0, seed=51)
    T = float(t[-1]) + 1.0
    a0, eta, beta = -1.0, 0.4, 1.5
    soe = core.hawkes_ll_soe(_buf(t), T, math.exp(a0), eta, np.array([beta]), np.array([beta]))
    ref = float(_ll_exp_const(t, T, a0, eta, beta))
    assert np.isclose(soe, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("seed", [3, 9])
def test_hawkes_ll_soe_matches_bruteforce(seed):
    # the O(M*n) recursion must match the O(n^2) definition
    rng = np.random.RandomState(seed)
    t = _event_times(150, rate=2.0, seed=seed)
    T = float(t[-1]) + 1.0
    w = rng.uniform(0.05, 0.5, size=3)
    beta = rng.uniform(0.5, 5.0, size=3)
    nu, eta = 0.4, 0.3
    got = core.hawkes_ll_soe(_buf(t), T, nu, eta, w, beta)
    ref = _soe_ll_reference(t, T, nu, eta, w, beta)
    assert np.isclose(got, ref, rtol=1e-8, atol=1e-6)


# --- SoE fit of the Lomax kernel, task #73 ----------------------------


@pytest.mark.parametrize("alpha,c", [(1.5, 1.0), (2.5, 1.0), (4.0, 2.0)])
def test_soe_fit_lomax_kernel_accuracy(alpha, c):
    # the SoE must approximate the (completely monotone) Lomax kernel
    # over [0, horizon]; the fit reports its own verified error bound,
    # which an independent re-check on a fresh grid must confirm
    from morie.tps_hawkes_jit import soe_fit_lomax

    horizon = 200.0
    w, beta, err = soe_fit_lomax(alpha, c, horizon, tol=1e-8)
    assert err <= 1e-8
    u = np.geomspace(horizon * 1e-3, horizon, 500)
    g_true = (alpha - 1.0) / c * (1.0 + u / c) ** (-alpha)
    g_soe = (w[None, :] * np.exp(-np.outer(u, beta))).sum(axis=1)
    assert np.max(np.abs(g_soe - g_true) / g_true) < 1e-6


@pytest.mark.parametrize("alpha,c", [(1.5, 1.0), (2.5, 1.0), (4.0, 2.0)])
def test_soe_fit_lomax_likelihood_matches_exact(alpha, c):
    # hawkes_ll_soe with the fitted Lomax SoE must reproduce the exact
    # O(n^2) Lomax likelihood -- SoE perturbs the LL only at fit error
    import math

    from morie.tps_hawkes_jit import soe_fit_lomax

    t = _event_times(400, rate=2.0, seed=17)
    T = float(t[-1]) + 1.0
    a0, eta = -1.0, 0.4
    exact = core.hawkes_ll_lomax_const(_buf(t), T, a0, eta, alpha, c)
    w, beta, err = soe_fit_lomax(alpha, c, T, tol=1e-8)
    soe = core.hawkes_ll_soe(_buf(t), T, math.exp(a0), eta, w, beta)
    assert np.isclose(soe, exact, rtol=1e-7)


def test_neg_loglik_jit_lomax_routes_through_soe():
    # at large n the lomax constant-baseline path routes through the
    # O(M*n) SoE engine; the result must still match the exact kernel
    from morie.tps_hawkes_jit import neg_loglik_jit

    t = _event_times(1500, rate=2.0, seed=23)  # n >= _SOE_MIN_N
    T = float(t[-1]) + 1.0
    a0, eta, alpha, c = -1.0, 0.4, 2.5, 1.0
    theta = np.array([a0, eta, alpha, c])
    routed = neg_loglik_jit(theta, t, T, "lomax", "constant")
    exact = core.hawkes_ll_lomax_const(_buf(t), T, a0, eta, alpha, c)
    assert np.isclose(routed, exact, rtol=1e-6)


def test_neg_loglik_jit_lomax_small_n_stays_exact():
    # below the crossover the exact O(n^2) kernel is used verbatim
    from morie.tps_hawkes_jit import neg_loglik_jit

    t = _event_times(200, rate=2.0, seed=23)
    T = float(t[-1]) + 1.0
    a0, eta, alpha, c = -1.0, 0.4, 2.5, 1.0
    theta = np.array([a0, eta, alpha, c])
    routed = neg_loglik_jit(theta, t, T, "lomax", "constant")
    exact = core.hawkes_ll_lomax_const(_buf(t), T, a0, eta, alpha, c)
    assert routed == exact  # same code path, bit-identical


# --- matrix-pencil exponential fitter, task #73 (gamma hybrid) --------


def test_soe_fit_matrix_pencil_recovers_known_soe():
    # the pencil must recover an exact 3-term SoE from its samples
    from morie.tps_hawkes_jit import _soe_fit_matrix_pencil

    beta_true = np.array([0.3, 1.2, 4.0])
    r_true = np.array([1.0, 0.6, 0.25])
    dt = 0.05
    k = np.arange(80)
    y = (r_true[None, :] * np.exp(-beta_true[None, :] * k[:, None] * dt)).sum(axis=1)
    beta, res = _soe_fit_matrix_pencil(y, dt)
    beta = np.asarray(beta, dtype=np.complex128)
    res = np.asarray(res, dtype=np.complex128)
    assert np.max(np.abs(beta.imag)) < 1e-6  # all poles real
    order = np.argsort(beta.real)
    assert np.allclose(beta.real[order], beta_true, atol=1e-6)
    assert np.allclose(res.real[order], r_true, atol=1e-6)


# --- complex-pole SoE engine, task #73 (gamma hybrid) -----------------
#
# The kernel exposes each complex entry point twice: hawkes_ll_soe_cplx /
# hawkes_ll_gamma_hybrid take complex128 arrays, and the _ri variants take
# the real and imaginary parts as separate float64 arrays. Only the _ri
# pair is reachable from morie: a complex128 argument has to arrive as a
# buffer with format "Zd", and nothing in the standard library produces
# one, so the complex overloads are callable only with numpy in hand.
# tps_hawkes_jit routes through the _ri entry points for exactly this
# reason, so these tests drive the same ones production does.


def _ri(seq):
    """Split a sequence of complex numbers into two float64 buffers."""
    import array as _pyarray

    vals = [complex(v) for v in seq]
    return (_pyarray.array("d", [v.real for v in vals]),
            _pyarray.array("d", [v.imag for v in vals]))


def _soe_ll_reference_cplx(t, T, nu, eta, w, beta):
    """Brute-force O(n^2) reference for the complex-pole SoE Hawkes
    likelihood: g(u) = Re(sum_m w[m]*exp(-beta[m]*u))."""
    import cmath
    import math

    w = [complex(v) for v in w]
    beta = [complex(v) for v in beta]
    tv = [float(v) for v in t]
    n = len(tv)
    log_sum = 0.0
    for i in range(n):
        s = 0.0
        for j in range(i):
            d = tv[i] - tv[j]
            s += sum(wm * cmath.exp(-bm * d)
                     for wm, bm in zip(w, beta)).real
        log_sum += math.log(nu + eta * s)
    integral = nu * T
    for i in range(n):
        u = T - tv[i]
        integral += eta * sum((wm / bm) * (1.0 - cmath.exp(-bm * u))
                              for wm, bm in zip(w, beta)).real
    return -(log_sum - integral)


def test_hawkes_ll_soe_cplx_reduces_to_real():
    # with purely real poles the complex engine equals the real one
    t = _event_times(300, rate=2.0, seed=61)
    T = float(t[-1]) + 1.0
    w = np.array([0.5, 0.3, 0.15])
    beta = np.array([0.4, 1.5, 4.0])
    real = core.hawkes_ll_soe(_buf(t), T, 0.4, 0.3, w, beta)
    w_re, w_im = _ri(w)
    b_re, b_im = _ri(beta)
    cplx = core.hawkes_ll_soe_cplx_ri(_buf(t), T, 0.4, 0.3, w_re, w_im,
                                      b_re, b_im)
    assert np.isclose(cplx, real, rtol=1e-12, atol=1e-9)


def test_hawkes_ll_soe_cplx_conjugate_pair_matches_bruteforce():
    # a real pole plus one conjugate pair, engine vs O(n^2) definition
    t = _event_times(150, rate=2.0, seed=63)
    T = float(t[-1]) + 1.0
    nu, eta = 0.4, 0.3
    w = [0.40 + 0j, 0.10 + 0.05j, 0.10 - 0.05j]
    beta = [0.70 + 0j, 1.20 + 0.80j, 1.20 - 0.80j]
    w_re, w_im = _ri(w)
    b_re, b_im = _ri(beta)
    got = core.hawkes_ll_soe_cplx_ri(_buf(t), T, nu, eta, w_re, w_im,
                                     b_re, b_im)
    ref = _soe_ll_reference_cplx(t, T, nu, eta, w, beta)
    assert np.isclose(got, ref, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("alpha,beta", [(1.5, 1.0), (2.5, 1.0), (4.0, 2.0), (1.2, 0.5)])
def test_soe_fit_gamma_tail_accuracy(alpha, beta):
    # the matrix-pencil SoE must reproduce the gamma tail past u_split,
    # with every mode decaying and the conjugate-paired sum real
    from morie.tps_hawkes_jit import soe_fit_gamma_tail

    u_split = 2.0 * (alpha - 1.0) / beta  # 2 x the peak
    import cmath

    w, beta_soe, err = soe_fit_gamma_tail(alpha, beta, u_split)
    # soe_fit_gamma_tail returns plain lists of complex, which is what
    # tps_hawkes_jit consumes; keep them that way rather than routing
    # through an array type that has no complex dtype.
    w = [complex(v) for v in w]
    beta_soe = [complex(v) for v in beta_soe]
    assert err < 1e-5
    assert all(b.real > 0.0 for b in beta_soe)  # all modes decay
    s = [0.0, 1.0 / beta, 5.0 / beta]
    val = [sum(wm * cmath.exp(-bm * u) for wm, bm in zip(w, beta_soe))
           for u in s]
    # Tolerance: ~1e-7 relative. Windows BLAS/LAPACK gives slightly looser
    # complex-arithmetic rounding than Linux/macOS (observed ~3e-9 absolute
    # imaginary residue on Windows vs ~7e-10 on Linux for the same inputs).
    assert max(abs(v.imag) for v in val) < \
        1e-7 * max(abs(v.real) for v in val) + 1e-10


@pytest.mark.parametrize("alpha,beta", [(1.5, 1.0), (2.5, 1.0), (4.0, 2.0)])
def test_hawkes_ll_gamma_hybrid_matches_exact(alpha, beta):
    # the hybrid (exact peak window + SoE tail) must reproduce the
    # exact O(n^2) gamma likelihood within the tail-fit tolerance
    from morie.tps_hawkes_jit import soe_fit_gamma_tail

    t = _event_times(400, rate=2.0, seed=29)
    T = float(t[-1]) + 1.0
    a0, eta = -1.0, 0.4
    u_split = 2.0 * (alpha - 1.0) / beta
    w, beta_soe, err = soe_fit_gamma_tail(alpha, beta, u_split)
    exact = core.hawkes_ll_gamma_const(_buf(t), T, a0, eta, alpha, beta)
    w_re, w_im = _ri(w)
    b_re, b_im = _ri(beta_soe)
    hybrid = core.hawkes_ll_gamma_hybrid_ri(
        _buf(t), T, a0, eta, alpha, beta, u_split, w_re, w_im, b_re, b_im)
    assert np.isclose(hybrid, exact, rtol=1e-4)


def test_neg_loglik_jit_gamma_routes_through_hybrid():
    # large n + slow decay -> the gamma path routes through the hybrid;
    # the result must still match the exact O(n^2) gamma likelihood
    from morie.tps_hawkes_jit import _gamma_trunc_cutoff, neg_loglik_jit

    t = _event_times(1500, rate=2.0, seed=31)
    T = float(t[-1]) + 1.0
    a0, eta, alpha, beta = -1.0, 0.4, 2.5, 0.4  # slow decay
    assert _gamma_trunc_cutoff(alpha, beta) >= t[-1] - t[0]  # hybrid route
    theta = np.array([a0, eta, alpha, beta])
    routed = neg_loglik_jit(theta, t, T, "gamma", "constant")
    exact = core.hawkes_ll_gamma_const(_buf(t), T, a0, eta, alpha, beta)
    assert np.isclose(routed, exact, rtol=1e-4)


def test_neg_loglik_jit_gamma_fast_decay_stays_exact():
    # fast decay -> truncation path, bit-identical to the exact kernel
    from morie.tps_hawkes_jit import neg_loglik_jit

    t = _event_times(1500, rate=2.0, seed=31)
    T = float(t[-1]) + 1.0
    a0, eta, alpha, beta = -1.0, 0.4, 2.5, 5.0  # fast decay
    theta = np.array([a0, eta, alpha, beta])
    routed = neg_loglik_jit(theta, t, T, "gamma", "constant")
    exact = core.hawkes_ll_gamma_const(_buf(t), T, a0, eta, alpha, beta)
    assert routed == exact
