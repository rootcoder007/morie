"""morie.tps_hawkes_advanced -- non-stationary Hawkes with non-exponential kernels.

Implements the Kwan-Chen-Dunsmuir (2024, arXiv:2408.09710v1) methodology
for Hawkes process likelihood inference when the baseline intensity is
time-varying *and* the excitation kernel is non-exponential (so the
intensity process is non-Markovian).

Companion to ``morie.tps_stochastic`` -- that module contains the
classical exponential-kernel constant-baseline (Markovian / Mohler 2011)
fit; this module adds:

  • Gamma, Weibull, and power-law (Lomax) excitation kernels.
  • Sinusoidal-with-trend time-varying baseline ν(t).
  • Time-rescaling residuals (Brown et al. 2002) for goodness-of-fit
    via Kolmogorov-Smirnov + Q-Q plots.
  • Eight-way model comparison via AIC and an explicit
    Markovian-vs-non-Markovian likelihood-ratio surface.

The complete intensity function is

    λ(t) = ν(t) + ∫_0^{t-} g(t - s) dN_s

with kernel decomposition g(u) = η · g̃(u) where η ∈ (0, 1) is the
branching ratio (mean offspring per event) and g̃ is a probability
density on [0, ∞).  Stationarity requires η < 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as sps
from morie.fn._sci_core import gammainc, gammaln, minimize  # noqa: F401 -- used inline

from .fn._richresult import RichResult
from .tps_hawkes_jit import has_jit_path, neg_loglik_jit

PROJECT = Path(__file__).resolve().parents[5]
FIG_OUT = PROJECT / "data/manifest/outputs/figures/tps_hawkes_advanced"

KernelKind = Literal["exponential", "gamma", "weibull", "lomax"]
BaselineKind = Literal["constant", "sinusoidal"]

KERNELS: tuple[KernelKind, ...] = ("exponential", "gamma", "weibull", "lomax")
BASELINES: tuple[BaselineKind, ...] = ("constant", "sinusoidal")


def _ensure_dirs() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)


# ── Kernel densities and CDFs ───────────────────────────────────────


def _kernel_density(u: np.ndarray, kind: KernelKind, psi: tuple[float, ...]) -> np.ndarray:
    """Normalised excitation density g̃(u) at lags u ≥ 0.

    Parameters
    ----------
    kind:
        One of ``"exponential"``, ``"gamma"``, ``"weibull"``, ``"lomax"``.
    psi:
        Kernel-specific parameter tuple -- see ``_n_kernel_params``.
    """
    u = np.asarray(u, dtype=float)
    if kind == "exponential":
        (beta,) = psi
        return beta * np.exp(-beta * u)
    if kind == "gamma":
        alpha, beta = psi
        # density: β^α u^{α-1} e^{-β u} / Γ(α)
        log_d = alpha * math.log(beta) + (alpha - 1) * np.log(np.maximum(u, 1e-300)) - beta * u - math.lgamma(alpha)
        return np.exp(log_d)
    if kind == "weibull":
        alpha, lam = psi
        x = u / lam
        return (alpha / lam) * np.power(np.maximum(x, 1e-300), alpha - 1) * np.exp(-np.power(x, alpha))
    if kind == "lomax":
        # v0.9.5.6+: scipy.stats.lomax convention.
        # density: alpha * c^alpha * (u+c)^{-(alpha+1)} for alpha > 0, c > 0.
        # Pre-v0.9.5.6 used the (alpha-1) shifted form which gave
        # zero density at alpha = 1 and required alpha > 1.
        alpha, c = psi
        log_d = math.log(alpha) + alpha * math.log(c) - (alpha + 1.0) * np.log(u + c)
        return np.exp(log_d)
    raise ValueError(f"unknown kernel kind: {kind}")


def _kernel_cdf(u: np.ndarray, kind: KernelKind, psi: tuple[float, ...]) -> np.ndarray:
    """Cumulative ∫_0^u g̃(v) dv, used to evaluate ∫_{t_i}^T g(t-t_i) dt."""
    u = np.asarray(u, dtype=float)
    if kind == "exponential":
        (beta,) = psi
        return 1.0 - np.exp(-beta * u)
    if kind == "gamma":
        alpha, beta = psi
        return gammainc(alpha, beta * u)
    if kind == "weibull":
        alpha, lam = psi
        return 1.0 - np.exp(-np.power(u / lam, alpha))
    if kind == "lomax":
        # v0.9.5.6+: scipy CDF 1 - (c/(u+c))^alpha (was: alpha-1).
        alpha, c = psi
        return 1.0 - np.power(c / (u + c), alpha)
    raise ValueError(f"unknown kernel kind: {kind}")


def _n_kernel_params(kind: KernelKind) -> int:
    return 1 if kind == "exponential" else 2


# ── Baselines and their integrals ───────────────────────────────────


def _baseline(t: np.ndarray, kind: BaselineKind, alpha: tuple[float, ...], T: float) -> np.ndarray:
    """Baseline intensity ν(t) on [0, T] in events / day.

    ``"constant"``  ->  ν(t) = exp(a₀)  (one parameter, log-link).
    ``"sinusoidal"``->  ν(t) = exp(a₀ + a₁·(t/T) + a₂ sin(2πt/365.25)
                                       + a₃ cos(2πt/365.25)).
    """
    t = np.asarray(t, dtype=float)
    if kind == "constant":
        (a0,) = alpha
        return np.full_like(t, math.exp(a0))
    if kind == "sinusoidal":
        a0, a1, a2, a3 = alpha
        return np.exp(
            a0 + a1 * (t / max(T, 1.0)) + a2 * np.sin(2 * math.pi * t / 365.25) + a3 * np.cos(2 * math.pi * t / 365.25)
        )
    raise ValueError(f"unknown baseline kind: {kind}")


def _baseline_integral(T: float, kind: BaselineKind, alpha: tuple[float, ...]) -> float:
    """∫_0^T ν(t) dt -- closed form for constant, trapezoidal otherwise."""
    if kind == "constant":
        (a0,) = alpha
        return math.exp(a0) * T
    grid = np.linspace(0.0, T, max(64, int(T) + 1))
    vals = _baseline(grid, kind, alpha, T)
    # NumPy 2.x renamed np.trapz -> np.trapezoid; fall back for older.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(vals, grid))


def _n_baseline_params(kind: BaselineKind) -> int:
    return 1 if kind == "constant" else 4


# ── Negative log-likelihood ─────────────────────────────────────────


def _split_theta(
    theta: np.ndarray, kernel_kind: KernelKind, baseline_kind: BaselineKind
) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    """θ -> (α-baseline, η-branching-ratio, ψ-kernel)."""
    nb = _n_baseline_params(baseline_kind)
    nk = _n_kernel_params(kernel_kind)
    if theta.size != nb + 1 + nk:
        raise ValueError(f"expected {nb + 1 + nk} params, got {theta.size}")
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1 :])
    return a, eta, psi


def _neg_loglik_general(
    theta: np.ndarray, t: np.ndarray, T: float, kernel_kind: KernelKind, baseline_kind: BaselineKind
) -> float:
    """Negative log-likelihood for a general non-stationary Hawkes process.

    Uses the form

        ℓ(θ) = Σᵢ log λ(tᵢ) − ∫_0^T λ(s) ds.

    With kernel decomposition g = η · g̃ and CDF F̃,

        ∫_0^T λ(s) ds = ∫_0^T ν(s) ds + η Σᵢ F̃(T − tᵢ).

    The intensity at events is computed by direct O(n²) summation --
    correct for n ≲ 5 000.  For n ≫ 10⁴ a recursive update is needed
    only in the exponential-kernel case (other kernels lack the
    memorylessness required for O(n) recursion).
    """
    # Fast path: Numba-JIT'd implementation with O(n) recursion for the
    # exponential kernel and O(n²) JIT'd loops for non-Markovian kernels.
    # See tps_hawkes_jit.has_jit_path for the supported combinations.
    if has_jit_path(kernel_kind, baseline_kind):
        return neg_loglik_jit(theta, t, T, kernel_kind, baseline_kind)

    nb = _n_baseline_params(baseline_kind)
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1 :])

    # box constraints
    if eta <= 1e-6 or eta >= 0.999:
        return 1e12
    if any(p <= 1e-6 for p in psi):
        return 1e12
    if kernel_kind == "lomax" and psi[0] <= 1.001:
        return 1e12  # need α > 1 for finite mean (scipy convention)

    # Σᵢ log λ(tᵢ)
    n = t.size
    nu_at_t = _baseline(t, baseline_kind, a, T)
    log_sum = 0.0
    for i in range(n):
        if i == 0:
            lam_i = nu_at_t[0]
        else:
            lags = t[i] - t[:i]
            lam_i = nu_at_t[i] + eta * np.sum(_kernel_density(lags, kernel_kind, psi))
        if lam_i <= 0:
            return 1e12
        log_sum += math.log(lam_i)

    # ∫_0^T λ(s) ds
    integral = _baseline_integral(T, baseline_kind, a) + eta * float(np.sum(_kernel_cdf(T - t, kernel_kind, psi)))
    return -(log_sum - integral)


# ── Initial-guess heuristics ────────────────────────────────────────


def _x0(kernel_kind: KernelKind, baseline_kind: BaselineKind, n: int, T: float, mean_dt: float) -> np.ndarray:
    rate = max(n / T, 1e-3)
    a = [math.log(rate * 0.6)] if baseline_kind == "constant" else [math.log(rate * 0.6), 0.0, 0.0, 0.0]
    eta = [0.4]
    if kernel_kind == "exponential":
        psi = [1.0 / max(mean_dt, 1e-3)]
    elif kernel_kind == "gamma":
        psi = [1.5, 1.0 / max(mean_dt, 1e-3)]
    elif kernel_kind == "weibull":
        psi = [1.5, max(mean_dt, 1e-3) * 1.2]
    else:  # lomax
        psi = [2.5, max(mean_dt, 1e-3) * 5.0]
    return np.array(a + eta + psi, dtype=float)


# ── Public fit + GoF ────────────────────────────────────────────────


_KIND_CODE = {"exponential": 0, "weibull": 1, "gamma": 2, "lomax": 3}
_BASELINE_CODE = {"constant": 0, "sinusoidal": 1}
HAWKES_METHODS = ("auto", "exact", "soe", "truncate", "em", "inar")


def _resolve_method(method: str, kernel_kind: str) -> str:
    if method not in HAWKES_METHODS:
        raise ValueError(f"method must be one of {', '.join(HAWKES_METHODS)}; got {method!r}")
    if method == "auto":
        # Weibull's exact window is where the kernel underflows: for shape < 1 that is the whole
        # record (O(n^2)), while its tail mass passes eps within days
        return {"exponential": "exact", "weibull": "truncate"}.get(kernel_kind, "soe")
    if method == "soe" and kernel_kind not in ("lomax", "gamma"):
        raise ValueError(
            "method='soe' applies to completely monotone kernels: 'lomax', and 'gamma' with shape < 1 "
            "(a gamma kernel with shape >= 1 is truncated at eps); use 'exact' or 'truncate'"
        )
    return method


def _soe_window(t, T, kernel_kind):
    """The sum-of-exponentials window: r = (u + c)/R in [delta, 1] for Lomax (c <= 100 in the fit),
    r = u/R from the smallest gap for gamma -- as rmoriebricklayer's core_hawkes_fit."""
    if kernel_kind == "lomax":
        return float(T) + 100.0, 1e-3 / (float(T) + 100.0)
    tl = [float(v) for v in t]
    gaps = [b - a for a, b in zip(tl[:-1], tl[1:]) if b > a]
    return float(T), max(min(gaps) if gaps else 1e-9, 1e-12) / float(T)


def _core_fit(t, T, kernel_kind, baseline_kind, method, eps, x0, bounds):
    """The whole fit in the compiled core (projected BFGS on the analytic gradient,
    morie::core::hawkes_fit_pbfgs): (theta, nll), or None without the core. The same routine
    rmoriebricklayer's core_hawkes_fit calls, so both arms reach the same optimum."""
    from .tps_hawkes_jit import HAS_CORE, _f64

    if not HAS_CORE:
        return None
    from .tps_hawkes_jit import _core_ext

    if not hasattr(_core_ext, "hawkes_fit_pbfgs"):
        return None
    R, delta = _soe_window(t, T, kernel_kind)
    code = {"exact": 0, "soe": 1, "truncate": 2}[method]
    th, f, _ = _core_ext.hawkes_fit_pbfgs(
        _f64(t),
        float(T),
        _BASELINE_CODE[baseline_kind],
        _KIND_CODE[kernel_kind],
        code,
        float(eps),
        R,
        delta,
        [float(b[0]) for b in bounds],
        [float(b[1]) for b in bounds],
        [float(v) for v in (x0._flat() if hasattr(x0, "_flat") else x0)],
        2000,
        1e-6,
    )
    return list(th), float(f)


def _core_objective(t, T, kernel_kind, baseline_kind, method, eps):
    """(nll, gradient) from the compiled core (morie::core::hawkes_nll_grad), or None without it."""
    from .tps_hawkes_jit import HAS_CORE, _f64

    if not HAS_CORE:
        return None
    from .tps_hawkes_jit import _core_ext

    if not hasattr(_core_ext, "hawkes_nll_grad"):
        return None
    tb = _f64(t)
    nb = _n_baseline_params(baseline_kind)
    kind, bk = _KIND_CODE[kernel_kind], _BASELINE_CODE[baseline_kind]
    code = {"exact": 0, "soe": 1, "truncate": 2}[method]
    soe_R, soe_delta = _soe_window(t, T, kernel_kind)

    def f(theta):
        th = [float(v) for v in np.asarray(theta)._flat()] if hasattr(np.asarray(theta), "_flat") else list(theta)
        nll, g = _core_ext.hawkes_nll_grad(
            tb, float(T), bk, th[:nb], th[nb], kind, th[nb + 1 :], code, float(eps), soe_R, soe_delta, True
        )
        if not math.isfinite(nll) or nll >= 1e11:
            return 1e12, [0.0] * len(th)
        return nll, list(g)

    return f


def _warm_start(t, T, kernel_kind, baseline_kind, bounds):
    """The exponential-kernel fit mapped into another kernel: Weibull (1, 1/beta) and gamma (1, beta)
    are the exponential kernel itself; Lomax (30, 30/beta) is its nearest member (Lomax tends to it
    as the shape grows with c = shape/beta)."""
    nb = _n_baseline_params(baseline_kind)
    n = int(np.asarray(t).size)
    mean_dt = float(np.mean(np.diff(np.asarray(t)))) if n > 1 else 1.0
    x0 = _x0("exponential", baseline_kind, n, T, mean_dt)
    eb = list(bounds[: nb + 1]) + [(0.1, 25.0)]
    fit = _core_fit(t, T, "exponential", baseline_kind, "exact", 1e-9, x0, eb)
    if fit is None:
        return None
    th = fit[0]
    beta = th[-1]
    psi = {"weibull": [1.0, 1.0 / beta], "gamma": [1.0, beta], "lomax": [30.0, 30.0 / beta]}[kernel_kind]
    start = th[:-1] + psi
    return [min(max(v, b[0]), b[1]) for v, b in zip(start, bounds)]


def fit_hawkes_general(
    t: np.ndarray,
    T: float,
    kernel_kind: KernelKind = "exponential",
    baseline_kind: BaselineKind = "constant",
    *,
    method: str = "auto",
    eps: float = 1e-9,
) -> dict:
    """MLE of a non-stationary Hawkes process.

    ``method`` chooses how the likelihood is evaluated (all with the analytic gradient):

    * ``"exact"``: Ozaki's O(n) recursion for the exponential kernel; for Weibull and gamma the
      double sum stops where the kernel underflows to exactly 0 (the same value as the full sum);
      Lomax the full O(n^2) sum.
    * ``"soe"``: the completely monotone kernels (Lomax; gamma with shape < 1) as a sum of
      exponentials (Beylkin & Monzon 2010), relative error ``eps`` per intensity, O(n K); a gamma
      kernel with shape >= 1 is truncated at ``eps``.
    * ``"truncate"``: each event excites only lags with kernel tail mass above ``eps`` (light tails).
    * ``"em"``: the EM algorithm (Veen & Schoenberg 2008): the same MLE, a different route.
    * ``"inar"``: Kirchner's (2017) INAR(p) least-squares estimator on binned counts, constant
      baseline only: a different (fast, approximate) estimator, also used for starting values.
    * ``"auto"`` (default): ``"exact"`` for the exponential kernel, ``"truncate"`` for Weibull,
      ``"soe"`` for Lomax and gamma (``"exact"`` without the compiled core).

    Returns a dict with ``theta``, ``nll``, ``aic``, ``bic``,
    ``branching_ratio``, ``baseline_params``, ``kernel_params``,
    and the time-rescaling KS statistic.
    """
    auto = method == "auto"
    method = _resolve_method(method, kernel_kind)
    # keep the events in [0, T]: one past T would give the kernel CDF a negative lag; one at T
    # (the last event, when T is the last event time) is part of the record
    t = np.asarray(t, dtype=float)
    t = t[(t >= 0.0) & (t <= T)]
    n = int(t.size)
    if n < 50:
        raise ValueError(f"too few events ({n}) for non-stationary fit")
    mean_dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
    x0 = _x0(kernel_kind, baseline_kind, n, T, mean_dt)

    # L-BFGS-B with explicit bounds avoids the Hawkes spike-train
    # degeneracy that Nelder-Mead drives into via the upper β wall.
    nb = _n_baseline_params(baseline_kind)
    bounds: list[tuple[float, float]] = []
    bounds += [(-15.0, 15.0)] + [(-5.0, 5.0)] * (nb - 1)  # a0, then a1..a3
    bounds += [(1e-3, 0.99)]  # eta
    if kernel_kind == "exponential":
        bounds += [(0.1, 25.0)]  # beta in [0.1, 25] /day
    elif kernel_kind == "weibull":
        bounds += [(0.1, 15.0), (1e-3, 100.0)]  # alpha, lambda
    elif kernel_kind == "gamma":
        bounds += [(0.1, 15.0), (0.05, 25.0)]  # alpha, beta
    elif kernel_kind == "lomax":
        bounds += [(1.05, 30.0), (1e-3, 100.0)]  # alpha, c
    if method == "em":
        from .tps_hawkes_jit import hawkes_em

        theta, nll_em, n_iter = hawkes_em(t, T, kernel_kind, baseline_kind, x0, bounds)
        # EM converges linearly: its steps shrink long before it reaches the maximum, so a small
        # step is not a stop. Finish with the projected BFGS from EM's point, as
        # rmoriebricklayer's core_hawkes_fit; it only moves uphill. The likelihood route is the
        # kernel's own "auto" one (exact for exponential, truncate for Weibull, sum of
        # exponentials for Lomax and gamma): the full O(n^2) Lomax sum made this polish the
        # slowest part of the fit.
        pol = _core_fit(t, T, kernel_kind, baseline_kind, _resolve_method("auto", kernel_kind), eps, theta, bounds)
        if pol is not None and pol[1] <= nll_em:
            theta = pol[0]
        res = None
    elif method == "inar":
        from .tps_hawkes_jit import hawkes_inar

        theta = hawkes_inar(t, T, kernel_kind, baseline_kind, bounds)
        res = None
    else:
        best = _core_fit(t, T, kernel_kind, baseline_kind, method, eps, x0, bounds)
        if best is not None:
            warm = _warm_start(t, T, kernel_kind, baseline_kind, bounds) if kernel_kind != "exponential" else None
            if warm is not None:
                # a second, deterministic start at the exponential fit (the kernel's shape-1 member for
                # Weibull and gamma): the likelihood is multimodal, and a model that contains the
                # exponential one should not report a worse maximum than it
                other = _core_fit(t, T, kernel_kind, baseline_kind, method, eps, warm, bounds)
                if other[1] < best[1]:
                    best = other
            from types import SimpleNamespace

            res = SimpleNamespace(x=best[0], fun=best[1], success=True)
        elif method == "exact" or auto:
            method = "exact"
            res = minimize(
                _neg_loglik_general,
                x0,
                args=(t, T, kernel_kind, baseline_kind),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
        else:
            raise RuntimeError(
                f"method={method!r} needs morie's compiled core (morie._core), not built in this install"
            )
    theta = np.asarray(res.x) if res is not None else np.asarray(theta)
    exact_obj = _core_objective(t, T, kernel_kind, baseline_kind, "exact", eps)
    # the reported nll is always the exact likelihood at the estimate (AIC comparable across methods)
    nll = (
        float(exact_obj(theta)[0])
        if exact_obj is not None
        else float(_neg_loglik_general(theta, t, T, kernel_kind, baseline_kind))
    )
    a, eta, psi = _split_theta(theta, kernel_kind, baseline_kind)
    k = theta.size
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll

    # time-rescaling residuals (Brown et al. 2002)
    u = _time_rescaling_residuals(theta, t, T, kernel_kind, baseline_kind)
    ks = sps.kstest(u, "uniform")
    return {
        "theta": theta.tolist(),
        "baseline_params": list(a),
        "branching_ratio": eta,
        "kernel_params": list(psi),
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "n": n,
        "T_days": float(T),
        "k_params": int(k),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "rescaled_uniforms": u.tolist()[:1000],
        "kernel_kind": kernel_kind,
        "baseline_kind": baseline_kind,
        "method": method,
        "eps": float(eps) if method in ("soe", "truncate") else None,
        "converged": bool(res.success) if res is not None else True,
    }


def _time_rescaling_residuals(
    theta: np.ndarray, t: np.ndarray, T: float, kernel_kind: KernelKind, baseline_kind: BaselineKind
) -> np.ndarray:
    """Return U_i = 1 - exp(-(Λ(t_i) - Λ(t_{i-1}))) ∈ [0, 1].

    Under correct specification U_i ∼ Uniform(0,1) iid (Brown et al.
    Neural Comput. 14:325-346, 2002).
    """
    nb = _n_baseline_params(baseline_kind)
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1 :])

    from .tps_hawkes_jit import HAS_CORE, _f64

    if HAS_CORE:
        from .tps_hawkes_jit import _core_ext

        if hasattr(_core_ext, "hawkes_rescaled"):
            # O(n) / O(n w) in the compiled core (was an O(n^2) Python loop)
            return np.asarray(
                _core_ext.hawkes_rescaled(
                    _f64(t),
                    float(T),
                    _BASELINE_CODE[baseline_kind],
                    [float(v) for v in a],
                    eta,
                    _KIND_CODE[kernel_kind],
                    [float(v) for v in psi],
                )
            )

    n = t.size
    # Λ(t_i) = ∫_0^{t_i} ν(s) ds + η Σ_{j<i} F̃(t_i - t_j)
    bl_grid = np.linspace(0.0, float(T), max(256, int(T) + 1))
    bl_vals = _baseline(bl_grid, baseline_kind, a, T)
    cum_baseline = np.concatenate(([0.0], np.cumsum(0.5 * (bl_vals[1:] + bl_vals[:-1]) * np.diff(bl_grid))))

    def Lambda_at(ti: float) -> float:
        bl = float(np.interp(ti, bl_grid, cum_baseline))
        prior = t[t < ti]
        excite = eta * float(np.sum(_kernel_cdf(ti - prior, kernel_kind, psi)))
        return bl + excite

    inc = np.empty(n)
    prev = 0.0
    for i, ti in enumerate(t):
        cur = Lambda_at(float(ti))
        inc[i] = max(cur - prev, 1e-12)
        prev = cur
    return 1.0 - np.exp(-inc)


# ── Pretty wrappers (RichResult) ────────────────────────────────────


def splitmix_uniforms(n: int, seed: int) -> list[float]:
    """n uniforms on [0, 1) from splitmix64 (Steele, Lea & Flood 2014): the same numbers as
    ``rmoriebricklayer::core_uniforms(n, seed)``, so the R and Python arms draw identical values
    where their results must agree (the within-day jitter of tied dates, a subsample).

    Examples:
        >>> [round(u, 12) for u in splitmix_uniforms(2, 42)]
        [0.741564878772, 0.159910392877]
    """
    mask = (1 << 64) - 1
    x = int(seed) & mask
    out = []
    for _ in range(int(n)):
        x = (x + 0x9E3779B97F4A7C15) & mask
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        out.append((z >> 11) * (1.0 / 9007199254740992.0))
    return out


def _events_to_days(df: pd.DataFrame, max_n: int | None) -> tuple[np.ndarray, float]:
    """Convert TPS event timestamps to a clean days-since-t0 vector.

    TPS open-data is daily-resolution (OCC_DATE has no hour), which
    creates many ties.  We jitter ties uniformly within the day so the
    kernel-at-zero density does not blow up; this is a standard
    workaround in temporal point-process estimation when only the day
    is observed.  Sub-sample to ``max_n`` for tractable O(n²) MLE.
    """
    from .tps_stochastic import _date_series

    dt = _date_series(df)
    # a random subsample thins the process (its clustering is lost: Hawkes fits on thinned data
    # flatten the kernel comparison); None keeps every event, now that the likelihood is fast
    vals = list(dt.tolist())
    if max_n is not None and len(vals) > max_n:
        # the max_n events with the smallest splitmix keys (seed 43), in data order: the same
        # subsample as rmorie's (a random sample in each language would differ)
        keys = splitmix_uniforms(len(vals), 43)
        keep = sorted(sorted(range(len(vals)), key=keys.__getitem__)[:max_n])
        vals = [vals[i] for i in keep]
    vals = sorted(vals)  # stable: ties keep data order
    t0 = vals[0]
    # U(0,1)-day jitter (splitmix64, seed 42, the same numbers as rmorie's) breaks the daily ties;
    # the order of days is kept because the jitter is below one day
    jit = splitmix_uniforms(len(vals), 42)
    t = sorted((v - t0).total_seconds() / 86400.0 + u for v, u in zip(vals, jit))
    t = np.asarray(t, dtype=float)
    return t, float(t[-1])


def hawkes_advanced_fit(
    df: pd.DataFrame,
    *,
    kernel: KernelKind = "gamma",
    baseline: BaselineKind = "sinusoidal",
    ds_name: str = "?",
    max_n: int | None = None,
    method: str = "auto",
    eps: float = 1e-9,
) -> RichResult:
    """Fit a single (kernel, baseline) combination with figures.

    A sibling to ``morie.tps_stochastic.hawkes_temporal_fit`` which is
    locked to the (exponential, constant) Markovian special case.
    """
    from .tps_stochastic import _try_savefig

    if isinstance(df, dict):
        # a plain {column: values} mapping, as the R arm accepts a list
        df = pd.DataFrame(df)
    if not hasattr(df, "columns"):
        raise TypeError(f"df must be a data frame or a dict of columns, got {type(df).__name__}")
    if "OCC_DATE" not in df.columns and "REPORT_DATE" not in df.columns:
        return RichResult(
            title=f"Hawkes-{kernel}/{baseline} -- {ds_name}", warnings=["no OCC_DATE or REPORT_DATE column"]
        )
    t, T = _events_to_days(df, max_n)
    if t.size < 100:
        return RichResult(title=f"Hawkes-{kernel}/{baseline} -- {ds_name}", warnings=[f"only {t.size} timestamps"])

    result = fit_hawkes_general(t, T, kernel_kind=kernel, baseline_kind=baseline, method=method, eps=eps)

    # QQ figure
    fig_path = None
    try:
        from morie.fn import _plot_core as plt

        u = np.array(result["rescaled_uniforms"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sps.probplot(u, dist="uniform", plot=ax[0])
        ax[0].set_title(f"{ds_name} -- Q-Q vs Uniform (KS p = {result['ks_pvalue']:.3f})")
        # density vs empirical
        ax[1].hist(u, bins=30, color="#3584e4", alpha=0.7, density=True, label="empirical U_i")
        ax[1].axhline(1.0, color="#e66100", ls="--", label="Uniform(0,1)")
        ax[1].set_xlim(0, 1)
        ax[1].legend()
        ax[1].set_title(f"{kernel}/{baseline} kernel")
        fig.suptitle(f"Time-rescaling residuals -- {ds_name}")
        plt.tight_layout()
        fig_path = _try_savefig(f"hawkes_qq_{kernel}_{baseline}_{ds_name}.png", fig)
    except Exception:
        pass

    a = result["baseline_params"]
    psi = result["kernel_params"]
    eta = result["branching_ratio"]
    summary = [
        ("Events fitted", result["n"]),
        ("Time window (days)", round(result["T_days"], 1)),
        ("Kernel", kernel),
        ("Baseline", baseline),
        ("η (branching ratio)", round(eta, 4)),
        ("Stationary?", "Yes (η<1)" if eta < 1 else "EXPLOSIVE"),
        ("Kernel params (ψ)", [round(x, 4) for x in psi]),
        ("Baseline params (α)", [round(x, 4) for x in a]),
        ("Negative log-likelihood", round(result["nll"], 1)),
        ("AIC", round(result["aic"], 1)),
        ("BIC", round(result["bic"], 1)),
        ("Time-rescaling KS stat", round(result["ks_stat"], 4)),
        ("Time-rescaling KS p-value", round(result["ks_pvalue"], 4)),
    ]
    interp = (
        f"Branching ratio η = {eta:.3f} -> mean {eta:.2f} offspring "
        f"per event. "
        + ("Process is stationary (η < 1)." if eta < 1 else "Process is EXPLOSIVE (η ≥ 1).")
        + " Time-rescaling KS p = "
        + ("strong fit; " if result["ks_pvalue"] >= 0.05 else "fit is rejected; ")
        + f"residuals {'consistent with' if result['ks_pvalue'] >= 0.05 else 'depart from'} Uniform(0,1)."
    )
    payload = dict(result)
    payload["figure_path"] = fig_path
    return RichResult(
        title=f"Hawkes [{kernel} kernel, {baseline} baseline] -- {ds_name}",
        summary_lines=summary,
        interpretation=interp,
        payload=payload,
    )


def compare_hawkes_kernels(
    df: pd.DataFrame,
    *,
    ds_name: str = "?",
    max_n: int | None = None,
    baselines: tuple[BaselineKind, ...] = BASELINES,
    kernels: tuple[KernelKind, ...] = KERNELS,
    method: str = "auto",
    eps: float = 1e-9,
) -> RichResult:
    """Fit every (kernel, baseline) combination and rank by AIC.

    Mirrors Section 5 (numerical examples) of Kwan-Chen-Dunsmuir 2024:
    the Markovian classical Hawkes is the (exponential, constant) row;
    the non-Markovian non-stationary models are everything else.
    """
    if "OCC_DATE" not in df.columns and "REPORT_DATE" not in df.columns:
        return RichResult(title=f"Hawkes comparison -- {ds_name}", warnings=["no OCC_DATE or REPORT_DATE column"])
    t, T = _events_to_days(df, max_n)
    if t.size < 100:
        return RichResult(title=f"Hawkes comparison -- {ds_name}", warnings=[f"only {t.size} timestamps"])

    rows = []
    for k in kernels:
        for b in baselines:
            try:
                fit = fit_hawkes_general(t, T, kernel_kind=k, baseline_kind=b, method=method, eps=eps)
                rows.append(
                    {
                        "kernel": k,
                        "baseline": b,
                        "k_params": fit["k_params"],
                        "nll": round(fit["nll"], 1),
                        "aic": round(fit["aic"], 1),
                        "bic": round(fit["bic"], 1),
                        "branching_ratio": round(fit["branching_ratio"], 3),
                        "ks_pvalue": round(fit["ks_pvalue"], 4),
                        "markovian": (k == "exponential"),
                        "stationary_baseline": (b == "constant"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "kernel": k,
                        "baseline": b,
                        "error": str(exc),
                    }
                )

    fitted = [r for r in rows if "error" not in r]
    fitted.sort(key=lambda r: r["aic"])
    best = fitted[0] if fitted else None
    summary = [("Combinations fitted", len(fitted)), ("Combinations failed", len(rows) - len(fitted))]
    if best is not None:
        summary += [
            ("Best (lowest AIC)", f"{best['kernel']} / {best['baseline']}"),
            (
                "Δ AIC vs Markovian classical",
                round(
                    next(
                        (r["aic"] for r in fitted if r["kernel"] == "exponential" and r["baseline"] == "constant"),
                        float("nan"),
                    )
                    - best["aic"],
                    1,
                ),
            ),
        ]
    interp = (
        "Comparison of the eight (kernel × baseline) combinations. "
        "The classical Markovian Hawkes (exponential, constant) is "
        "the special case where the bivariate process (N_t, λ_t) is "
        "Markov. All non-exponential kernels and the time-varying "
        "sinusoidal baseline yield non-Markovian intensity processes; "
        "their large-sample theory is the contribution of "
        "Kwan-Chen-Dunsmuir (2024)."
    )
    return RichResult(
        title=f"Markovian vs non-Markovian Hawkes -- {ds_name}",
        summary_lines=summary,
        interpretation=interp,
        payload={"rows": rows, "best": best},
    )


def hawkes_markovian_vs_nonmarkovian(
    df: pd.DataFrame, *, ds_name: str = "?", max_n: int | None = None, method: str = "auto", eps: float = 1e-9
) -> RichResult:
    """Focused 2-way comparison: classical exp/const vs gamma/sinusoidal.

    The two endpoints of the Kwan-Chen-Dunsmuir framework -- quickest to
    run on the dashboard. ``max_n=None`` keeps every event; ``method`` and ``eps`` as in
    :func:`fit_hawkes_general`.
    """
    return compare_hawkes_kernels(
        df,
        ds_name=ds_name,
        max_n=max_n,
        kernels=("exponential", "gamma"),
        baselines=("constant", "sinusoidal"),
        method=method,
        eps=eps,
    )
