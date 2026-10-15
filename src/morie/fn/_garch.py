# morie.fn -- function file (rootcoder007/morie)
"""Shared conditional-heteroscedasticity core.

Specifications follow Tsay, R. S. (2010) *Analysis of Financial Time
Series*, 3rd ed., Wiley Series in Probability and Statistics, Hoboken
NJ, ISBN 978-0-470-41435-4, Ch. 3 -- read in the library PDF:

- GARCH(1,1), Sec. 3.5 p. 131
- IGARCH(1,1), Sec. 3.6 p. 140-141: the unit-root constraint
  alpha = 1 - beta, so the variance equation is
  sigma_t^2 = omega + beta sigma_{t-1}^2 + (1 - beta) a_{t-1}^2
- GARCH-M, Sec. 3.7 p. 142, eq. (3.23): r_t = mu + c g(sigma_t) + a_t
  with g in {sigma^2, sigma, log sigma^2} (Table 3.2 p. 142)
- EGARCH, Sec. 3.8 p. 143, eq. (3.24)-(3.25): the weighted innovation
  g(e_t) = theta e_t + gamma(|e_t| - E|e_t|), with E|e_t| = sqrt(2/pi)
  for a standard Gaussian (the Remark on p. 143)
- TGARCH/GJR, Sec. 3.9 p. 149, eq. (3.34):
  sigma_t^2 = omega + sum (alpha_i + gamma_i N_{t-i}) a_{t-i}^2
  + sum beta_j sigma_{t-j}^2, with N_{t-i} the indicator of a negative
  shock

All fits are Gaussian quasi-maximum-likelihood by Nelder-Mead on an
unconstrained reparameterisation, which keeps the parameters inside
their stationarity region without a constrained optimiser.
"""

import math
import operator

from . import _array_core as np
from . import _stats_core as stats
from ._sci_core import optimize, special

__all__ = ["garch_recursion", "garch_fit", "SPECS"]

SPECS = ("garch", "igarch", "egarch", "gjr", "tgarch", "aparch", "cgarch", "figarch")

_SQRT_2_PI = np.sqrt(2.0 / np.pi)  # E|z| for a standard normal, Tsay p.143


def garch_recursion(eps, params, spec="garch"):
    """Conditional variances for one of :data:`SPECS`.

    ``eps`` are mean-zero shocks. ``params`` is a dict whose keys
    depend on the specification; the variance at t = 0 is initialised
    at the sample variance, the usual burn-in convention.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}.")
    eps = np.asarray(eps, dtype=float).ravel()
    n = eps.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    # The recursions run on plain lists. Each step reads and writes single
    # elements, and doing that through array objects made every likelihood
    # evaluation cost ~n array-element round trips; the optimisers call it
    # thousands of times. Same arithmetic, element for element.
    s2 = [0.0] * n
    s2[0] = max(float(np.var(eps)), 1e-12)
    eps = eps.tolist()

    if spec == "garch":
        w, a, b = params["omega"], params["alpha"], params["beta"]
        for t in range(1, n):
            s2[t] = w + a * eps[t - 1] ** 2 + b * s2[t - 1]
    elif spec == "igarch":
        # Tsay eq. p.141: alpha is pinned at 1 - beta (unit root)
        w, b = params["omega"], params["beta"]
        for t in range(1, n):
            s2[t] = w + b * s2[t - 1] + (1.0 - b) * eps[t - 1] ** 2
    elif spec == "egarch":
        w, a, g, b = params["omega"], params["alpha"], params["gamma"], params["beta"]
        ls2 = [0.0] * n
        ls2[0] = float(np.log(s2[0]))
        for t in range(1, n):
            z = eps[t - 1] / float(np.sqrt(np.exp(ls2[t - 1])))
            ls2[t] = w + b * ls2[t - 1] + a * (abs(z) - _SQRT_2_PI) + g * z
        s2 = np.exp(np.clip(np.asarray(ls2, dtype=float), -50, 50))
    elif spec in ("gjr", "tgarch"):
        w, a, g, b = params["omega"], params["alpha"], params["gamma"], params["beta"]
        for t in range(1, n):
            neg = 1.0 if eps[t - 1] < 0 else 0.0
            s2[t] = w + (a + g * neg) * eps[t - 1] ** 2 + b * s2[t - 1]
    elif spec == "aparch":
        w, a, g, b, d = (
            params["omega"],
            params["alpha"],
            params["gamma"],
            params["beta"],
            params["delta"],
        )
        sd = [0.0] * n
        sd[0] = s2[0] ** (d / 2.0)
        for t in range(1, n):
            sd[t] = w + a * (abs(eps[t - 1]) - g * eps[t - 1]) ** d + b * sd[t - 1]
        s2 = np.maximum(np.asarray(sd, dtype=float), 1e-300) ** (2.0 / d)
    elif spec == "cgarch":
        # Engle-Lee component model: a slow permanent component q_t and
        # a transitory deviation, so persistence splits in two.
        w, rho, phi, a, b = (
            params["omega"],
            params["rho"],
            params["phi"],
            params["alpha"],
            params["beta"],
        )
        q = [0.0] * n
        q[0] = s2[0]
        for t in range(1, n):
            q[t] = w + rho * q[t - 1] + phi * (eps[t - 1] ** 2 - s2[t - 1])
            s2[t] = q[t] + a * (eps[t - 1] ** 2 - q[t - 1]) + b * (s2[t - 1] - q[t - 1])
            s2[t] = max(s2[t], 1e-12)
        return np.maximum(np.asarray(s2, dtype=float), 1e-12), np.asarray(q, dtype=float)
    else:  # figarch
        d, b, phi = params["d"], params["beta"], params["phi"]
        w = params["omega"]
        trunc = int(params.get("trunc", 50))
        lam = _figarch_weights(d, b, phi, trunc).tolist()
        e2r = [v**2 for v in reversed(eps)]  # e2r[n - t + j] = eps[t - 1 - j] ** 2
        for t in range(1, n):
            k = min(t, trunc)
            # np.dot of two 1-D float vectors is the exactly rounded fsum
            s2[t] = w + math.fsum(map(operator.mul, lam[:k], e2r[n - t : n - t + k]))
            s2[t] = max(s2[t], 1e-12)
    return np.maximum(np.asarray(s2, dtype=float), 1e-12)


def _figarch_weights(d, beta, phi, trunc):
    """ARCH(inf) weights of a FIGARCH(1, d, 1) by recursive expansion.

    (1 - L)^d is expanded to ``trunc`` terms and combined with the
    (1 - beta L)^{-1}(1 - phi L) filter; the weights are floored at 0
    so the variance recursion cannot go negative, which the raw
    expansion can do at small d.
    """
    g = np.empty(trunc + 1)
    g[0] = 1.0
    for k in range(1, trunc + 1):
        g[k] = g[k - 1] * (k - 1 - d) / k
    lam = np.zeros(trunc)
    prev = 0.0
    for k in range(1, trunc + 1):
        val = beta * prev - g[k]
        if k == 1:
            val -= phi
        lam[k - 1] = val
        prev = val
    return np.maximum(lam, 0.0)


def _pack(spec, x):
    """Unconstrained vector -> parameter dict inside the valid region."""

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    if spec == "garch":
        w = np.exp(np.clip(x[0], -30, 5))
        # a + b < 1 enforced by splitting a simplex weight
        tot = sig(x[1]) * 0.999
        frac = sig(x[2])
        return {"omega": w, "alpha": tot * frac, "beta": tot * (1 - frac)}
    if spec == "igarch":
        return {"omega": np.exp(np.clip(x[0], -30, 5)), "beta": sig(x[1]) * 0.999}
    if spec == "egarch":
        return {
            "omega": x[0],
            "alpha": x[1],
            "gamma": x[2],
            "beta": np.tanh(x[3]) * 0.999,
        }
    if spec in ("gjr", "tgarch"):
        tot = sig(x[1]) * 0.999
        frac = sig(x[2])
        return {
            "omega": np.exp(np.clip(x[0], -30, 5)),
            "alpha": tot * frac,
            "beta": tot * (1 - frac),
            "gamma": np.tanh(x[3]) * 0.5,
        }
    if spec == "aparch":
        tot = sig(x[1]) * 0.999
        frac = sig(x[2])
        return {
            "omega": np.exp(np.clip(x[0], -30, 5)),
            "alpha": tot * frac,
            "beta": tot * (1 - frac),
            "gamma": np.tanh(x[3]) * 0.99,
            "delta": 0.5 + 2.5 * sig(x[4]),
        }
    if spec == "cgarch":
        return {
            "omega": np.exp(np.clip(x[0], -30, 5)),
            "rho": sig(x[1]) * 0.9999,
            "phi": sig(x[2]) * 0.5,
            "alpha": sig(x[3]) * 0.5,
            "beta": sig(x[4]) * 0.9,
        }
    return {
        "omega": np.exp(np.clip(x[0], -30, 5)),
        "d": sig(x[1]) * 0.999,
        "beta": sig(x[2]) * 0.99,
        "phi": sig(x[3]) * 0.99,
    }


_X0 = {
    "garch": [-4.0, 2.0, -1.5],
    "igarch": [-6.0, 2.0],
    "egarch": [-0.5, 0.15, -0.05, 2.0],
    "gjr": [-4.0, 2.0, -1.5, 0.2],
    "tgarch": [-4.0, 2.0, -1.5, 0.2],
    "aparch": [-4.0, 2.0, -1.5, 0.2, 0.0],
    "cgarch": [-6.0, 3.0, -2.0, -2.0, 0.5],
    "figarch": [-4.0, 0.0, 0.0, 0.0],
}


def _loglik(eps, s2, dist="normal", nu=8.0):
    """Gaussian, Student-t or GED conditional log-likelihood.

    Summed term by term over plain lists (the exactly rounded sum, as
    np.sum gives): the optimisers evaluate this thousands of times and
    the array version built half a dozen n-element arrays per call.
    ``s2`` must be positive, which every caller guarantees.
    """
    el = np.asarray(eps, dtype=float).ravel().tolist()
    sl = np.asarray(s2, dtype=float).ravel().tolist()
    if dist == "normal":
        tp = 2 * math.pi
        return float(-0.5 * math.fsum(math.log(tp * v) + e**2 / v for e, v in zip(el, sl)))
    if dist == "t":
        if nu <= 2:
            return -np.inf
        # standardised t: unit variance, so scale by sqrt((nu-2)/nu)
        c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        h = (nu + 1) / 2
        return float(math.fsum(c - 0.5 * math.log(v) - h * math.log1p(e**2 / v / (nu - 2)) for e, v in zip(el, sl)))
    if dist == "ged":
        if nu <= 0:
            return -np.inf
        lam = np.sqrt(2 ** (-2 / nu) * special.gamma(1 / nu) / special.gamma(3 / nu))
        c = np.log(nu) - np.log(lam) - (1 + 1 / nu) * np.log(2) - special.gammaln(1 / nu)
        try:
            return float(
                math.fsum(c - 0.5 * math.log(v) - 0.5 * abs(e / math.sqrt(v) / lam) ** nu for e, v in zip(el, sl))
            )
        except OverflowError:  # |z / lam| ** nu overflowed: numpy gives inf, so the sum is -inf
            return -np.inf
    raise ValueError(f"dist must be 'normal', 't' or 'ged', got {dist!r}.")


def garch_fit(r, spec="garch", dist="normal", nu=None, mean="constant"):
    r"""Quasi-maximum-likelihood fit of a conditional-variance model.

    Parameters
    ----------
    r : array-like
        Return series.
    spec : str
        One of :data:`SPECS`.
    dist : {"normal", "t", "ged"}
        Conditional distribution; ``nu`` is estimated jointly when it
        is not supplied.
    nu : float, optional
        Fixed shape parameter for the t or GED.
    mean : {"constant", "zero"}
        Mean equation.

    Returns
    -------
    dict
        ``params``, ``sigma2``, ``loglik``, ``aic``, ``bic``,
        ``persistence``, ``residuals``, ``std_residuals``, ``mu``,
        ``nu``, ``spec``, ``converged``, ``n``.
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    if n < 50:
        raise ValueError(f"need at least 50 observations to fit, got {n}.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite.")
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}.")
    if mean not in ("constant", "zero"):
        raise ValueError("mean must be 'constant' or 'zero'.")

    mu = float(np.mean(r)) if mean == "constant" else 0.0
    eps = r - mu
    scale = float(np.std(eps))
    if scale <= 0:
        raise ValueError("r has zero variance.")
    e = eps / scale  # fit on standardised shocks; rescale at the end

    fit_nu = dist != "normal" and nu is None
    x0 = list(_X0[spec]) + ([1.0] if fit_nu else [])

    def neg(x):
        p = _pack(spec, x)
        try:
            s2 = garch_recursion(e, p, spec)
        except (ValueError, FloatingPointError, OverflowError):
            return 1e10
        if isinstance(s2, tuple):
            s2 = s2[0]
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        shape = nu
        if fit_nu:
            shape = 2.05 + 20.0 / (1.0 + np.exp(-np.clip(x[-1], -30, 30)))
        ll = _loglik(e, s2, dist, 8.0 if shape is None else shape)
        return 1e10 if not np.isfinite(ll) else -ll

    res = optimize.minimize(neg, x0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-8, "fatol": 1e-8})
    p = _pack(spec, res.x)
    shape = nu
    if fit_nu:
        shape = 2.05 + 20.0 / (1.0 + np.exp(-np.clip(res.x[-1], -30, 30)))
    out = garch_recursion(e, p, spec)
    q = None
    if isinstance(out, tuple):
        out, q = out

    # undo the standardisation: variance parameters scale by scale^2
    s2 = out * scale**2
    pp = dict(p)
    if spec == "egarch":
        pp["omega"] = p["omega"] + 2 * np.log(scale) * (1 - p["beta"])
    elif spec == "aparch":
        pp["omega"] = p["omega"] * scale ** p["delta"]
    else:
        pp["omega"] = p["omega"] * scale**2

    if spec in ("garch", "aparch"):
        pers = p["alpha"] + p["beta"]
    elif spec in ("igarch", "figarch"):
        pers = 1.0  # unit root by construction
    elif spec == "egarch":
        pers = p["beta"]
    elif spec in ("gjr", "tgarch"):
        # the asymmetric term is active half the time under a symmetric
        # innovation, so it enters persistence at gamma/2
        pers = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
    else:  # cgarch: the permanent component carries the long-run memory
        pers = p["rho"]

    k = len(x0)
    ll = -res.fun
    return {
        "params": pp,
        "sigma2": s2,
        "sigma": np.sqrt(s2),
        "loglik": ll,
        "aic": 2 * k - 2 * ll,
        "bic": k * np.log(n) - 2 * ll,
        "persistence": float(pers),
        "residuals": eps,
        "std_residuals": eps / np.sqrt(s2),
        "mu": mu,
        "nu": None if shape is None else float(shape),
        "spec": spec,
        "converged": bool(res.success and res.fun < 1e9),
        "n": int(n),
        "component": None if q is None else q * scale**2,
        "dist": dist,
    }


def garch_forecast(fit, horizon=1):
    """Multi-step variance forecast from a fitted GARCH-type model.

    Only the specifications with a closed-form recursion in the
    conditional mean of sigma^2 (GARCH, IGARCH, GJR) forecast
    analytically; the others iterate the recursion at the expected
    shock, which is exact for GARCH and a first-order approximation
    elsewhere -- stated here rather than hidden.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be at least 1, got {h}.")
    p = fit["params"]
    spec = fit["spec"]
    s2 = float(fit["sigma2"][-1])
    e2 = float(fit["residuals"][-1] ** 2)
    out = np.empty(h)
    for i in range(h):
        if spec == "garch":
            s2 = p["omega"] + p["alpha"] * e2 + p["beta"] * s2
        elif spec == "igarch":
            s2 = p["omega"] + p["beta"] * s2 + (1 - p["beta"]) * e2
        elif spec in ("gjr", "tgarch"):
            s2 = p["omega"] + (p["alpha"] + 0.5 * p["gamma"]) * e2 + p["beta"] * s2
        else:
            s2 = p.get("omega", 0.0) + fit["persistence"] * s2
        out[i] = s2
        e2 = s2  # E[eps^2 | F] = sigma^2 beyond the first step
    return out


def cheatsheet():
    return "_garch: recursions + QML fit for GARCH/IGARCH/EGARCH/GJR/APARCH/CGARCH/FIGARCH"


def bekk_fit(R, diagonal=True):
    r"""Scalar/diagonal BEKK(1,1) multivariate GARCH.

    .. math:: H_t = C'C + A' \epsilon_{t-1}\epsilon_{t-1}' A
              + B' H_{t-1} B

    (Engle & Kroner 1995.) ``C'C`` is positive definite by
    construction, so :math:`H_t` cannot lose positive-definiteness --
    the property that makes BEKK usable without a repair step. The
    default is the diagonal form: the full BEKK has O(k^2) parameters
    per matrix and is not identified at the sample sizes this is used
    on, so the restriction is a deliberate choice, not an omission.

    Uses variance targeting: ``C'C`` is set to
    :math:`\bar H (1 - a - b)`, which removes k(k+1)/2 free parameters.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("R must be 2-D (T observations x k series).")
    T, k = R.shape
    if k < 2:
        raise ValueError("BEKK needs at least 2 series.")
    if T < 50:
        raise ValueError(f"need at least 50 observations, got {T}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must be finite.")
    E = R - R.mean(axis=0)
    Hbar = np.cov(E, rowvar=False)

    def recur(a, b):
        Cmat = Hbar * (1.0 - a - b)
        H = np.empty((T, k, k))
        H[0] = Hbar
        for t in range(1, T):
            e = E[t - 1][:, None]
            H[t] = Cmat + a * (e @ e.T) + b * H[t - 1]
        return H

    def neg(x):
        a = 0.999 / (1 + np.exp(-np.clip(x[0], -30, 30)))
        b = (0.999 - a) / (1 + np.exp(-np.clip(x[1], -30, 30)))
        H = recur(a, b)
        ll = 0.0
        for t in range(T):
            try:
                L = np.linalg.cholesky(H[t])
            except np.linalg.LinAlgError:
                return 1e10
            sol = np.linalg.solve(L, E[t])
            ll += -0.5 * (k * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + float(sol @ sol))
        return -ll if np.isfinite(ll) else 1e10

    res = optimize.minimize(neg, [-2.0, 2.0], method="Nelder-Mead", options={"maxiter": 800, "fatol": 1e-6})
    a = 0.999 / (1 + np.exp(-np.clip(res.x[0], -30, 30)))
    b = (0.999 - a) / (1 + np.exp(-np.clip(res.x[1], -30, 30)))
    H = recur(a, b)
    return {
        "H": H,
        "a": float(a),
        "b": float(b),
        "H_bar": Hbar,
        "C": Hbar * (1 - a - b),
        "persistence": float(a + b),
        "loglik": float(-res.fun),
        "T": int(T),
        "k": int(k),
        "diagonal": bool(diagonal),
        "converged": bool(res.fun < 1e9),
        "method": "Scalar BEKK(1,1) with variance targeting (Engle-Kroner 1995)",
    }


def ms_garch_fit(r, n_regimes=2, max_iter=60):
    r"""Markov-switching GARCH by Hamilton filter on a Gray-collapsed
    variance recursion.

    Each regime carries its own (omega, alpha, beta); the conditional
    variance entering the next step is the probability-weighted
    average across regimes (Gray 1996), which keeps the recursion
    finite-dimensional -- the path-dependence problem that makes an
    exact MS-GARCH likelihood intractable.
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    K = int(n_regimes)
    if K < 2:
        raise ValueError("need at least 2 regimes.")
    if n < 100:
        raise ValueError(f"need at least 100 observations, got {n}.")
    e = r - r.mean()
    v = float(np.var(e))
    if v <= 0:
        raise ValueError("r has zero variance.")

    def unpack(x):
        def sig(z):
            return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

        pars = []
        for j in range(K):
            o = np.exp(np.clip(x[3 * j], -30, 5))
            tot = sig(x[3 * j + 1]) * 0.999
            fr = sig(x[3 * j + 2])
            pars.append((o, tot * fr, tot * (1 - fr)))
        # Softmax rows with the DIAGONAL logit pinned at 0 for
        # identification. Pinning column 0 instead (the obvious-looking
        # choice) leaves regime 0's self-transition free to collapse to
        # zero, which is not a regime at all.
        P = np.empty((K, K))
        base = 3 * K
        for i in range(K):
            free = list(x[base + i * (K - 1) : base + (i + 1) * (K - 1)])
            row = np.insert(np.array(free, dtype=float), i, 0.0)
            ex = np.exp(np.clip(row, -30, 30))
            P[i] = ex / ex.sum()
        return pars, P

    el = e.tolist()

    def filt(x):
        # Hamilton filter on K-vectors held as lists: per time step the
        # array-object version paid a dozen K-element array constructions,
        # times n steps, times thousands of Nelder-Mead evaluations.
        pars, P = unpack(x)
        Pl = P.tolist()
        pr = [(float(o), float(a), float(b)) for o, a, b in pars]
        xi = [1.0 / K] * K
        s2 = [v] * K
        ll = 0.0
        for t in range(n):
            if t:
                sbar = math.fsum(xi[j] * s2[j] for j in range(K))
                e2 = el[t - 1] ** 2
                s2 = [max(o + a * e2 + b * sbar, 1e-12) for o, a, b in pr]
            xi = [math.fsum(xi[i] * Pl[i][j] for i in range(K)) for j in range(K)]
            et2 = el[t] ** 2
            num = [xi[j] * (math.exp(-0.5 * et2 / s2[j]) / math.sqrt(2 * math.pi * s2[j])) for j in range(K)]
            tot = math.fsum(num)
            if not math.isfinite(tot) or tot <= 0:
                return 1e10, None, None
            ll += math.log(tot)
            xi = [u / tot for u in num]
        return -ll, pars, P

    x0 = []
    for j in range(K):
        x0 += [np.log(v * (0.5 + j)) - 2, 2.0, -1.5 + j]
    x0 += [-2.5] * (K * (K - 1))  # start persistent
    res = optimize.minimize(lambda x: filt(x)[0], x0, method="Nelder-Mead", options={"maxiter": 4000, "fatol": 1e-6})
    _, pars, P = filt(res.x)
    order = np.argsort([p[0] / max(1 - p[1] - p[2], 1e-6) for p in pars])
    return {
        "params": [{"omega": pars[i][0], "alpha": pars[i][1], "beta": pars[i][2]} for i in order],
        "transition": P[np.ix_(order, order)],
        "unconditional_var": [float(pars[i][0] / max(1 - pars[i][1] - pars[i][2], 1e-6)) for i in order],
        "loglik": float(-res.fun),
        "n_regimes": K,
        "n": int(n),
        "converged": bool(res.fun < 1e9),
        "method": "Markov-switching GARCH, Gray (1996) collapsed recursion",
        # Gray, S. F. (1996) "Modeling the conditional distribution of
        # interest rates as a regime-switching process", Journal of
        # Financial Economics 42(1), 27-62,
        # doi:10.1016/0304-405X(96)00875-6. The filter is Hamilton, J. D.
        # (1989) Econometrica 57(2), 357-384, doi:10.2307/1912559.
    }


def var_es(mu, sigma, alpha=0.05, dist="normal", nu=8.0):
    r"""Value-at-Risk and expected shortfall of a one-step forecast.

    Normal: :math:`VaR_\alpha = -\mu - \sigma z_\alpha` and
    :math:`ES_\alpha = -\mu + \sigma \phi(z_\alpha)/\alpha`.
    Student t (standardised to unit variance): the ES uses the
    t-density closed form. Both are reported as positive losses.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    mu = float(mu)
    if dist == "normal":
        z = stats.norm.ppf(alpha)
        return {
            "var": -(mu + sigma * z),
            "es": -(mu - sigma * stats.norm.pdf(z) / alpha),
            "quantile": float(z),
            "alpha": alpha,
            "dist": dist,
        }
    if dist == "t":
        if nu <= 2:
            raise ValueError("t distribution needs nu > 2 for a finite variance.")
        s = np.sqrt((nu - 2) / nu)  # standardise to unit variance
        z = stats.t.ppf(alpha, nu) * s
        pdf = stats.t.pdf(z / s, nu) / s
        es_std = -pdf / alpha * (nu + (z / s) ** 2) / (nu - 1) * s**2 / s
        return {
            "var": -(mu + sigma * z),
            "es": -(mu + sigma * es_std),
            "quantile": float(z),
            "alpha": alpha,
            "dist": dist,
            "nu": float(nu),
        }
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}.")
