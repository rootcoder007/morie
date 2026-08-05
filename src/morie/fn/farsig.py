# morie.fn -- function file (rootcoder007/morie)
"""Farrington flexible algorithm."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["farrington_signal"]


def _irls_qpois(X, y, w, iters=60, tol=1e-12):
    """Quasi-Poisson IRLS with prior weights w; returns (beta, mu, XtWXinv)."""
    n = len(y)
    p = len(X[0])
    b = [0.0] * p
    m = sum(y[i] * w[i] for i in range(n)) / max(sum(w), 1e-300)
    b[0] = math.log(m if m > 0.0 else 0.5)
    for _ in range(iters):
        eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
        mu = [math.exp(e) for e in eta]
        A = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        for i in range(n):
            wi = w[i] * mu[i]
            z = eta[i] + (y[i] - mu[i]) / mu[i] if mu[i] > 0.0 else eta[i]
            for a in range(p):
                for c in range(p):
                    A[a][c] += wi * X[i][a] * X[i][c]
                rhs[a] += wi * X[i][a] * z
        nb = core.cholsolve(A, rhs)
        d = max(abs(nb[j] - b[j]) for j in range(p))
        b = nb
        if d < tol:
            break
    eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
    mu = [math.exp(e) for e in eta]
    A = [[sum(w[i] * mu[i] * X[i][a] * X[i][c] for i in range(n))
          for c in range(p)] for a in range(p)]
    inv = [core.cholsolve(A, [1.0 if j == c else 0.0 for j in range(p)])
           for c in range(p)]
    return b, mu, inv


def farrington_signal(counts, baseline_years=5, reference_window=3,
                      period=52, alpha=0.005, reweight=True, trend=True):
    """
    Farrington flexible algorithm

    Formula: a quasi-Poisson GLM with overdispersion fitted to the
    reference baseline -- the same calendar window in each of the
    previous ``baseline_years`` years, plus/minus ``reference_window``
    time units -- and a 2/3-power upper threshold at the current point.

    With log mu_i = alpha + beta t_i, dispersion
    phi = max(1, sum w_i (y_i - mu_i)^2 / mu_i / (n - p)), prediction
    mu0 and its standard error se0 on the response scale, the threshold
    is (Farrington et al 1996 sec. 2.3)

        tau = phi + se0^2 / mu0
        se  = sqrt(4/9 mu0^(1/3) tau)
        U   = (mu0^(2/3) + z_{1-alpha/2} se)^(3/2)

    and an alarm is raised when the current count exceeds U.  When
    ``reweight`` is set, the fit is repeated with the Farrington weights
    derived from standardised Anscombe residuals,
    a_i = 1.5 (y^(2/3) mu^(-1/6) - mu^(1/2)) / sqrt(phi (1 - h_i)),
    down-weighting past outbreaks by a_i^-2.  Following Noufaily et al
    (2013) the trend term is retained whether or not it is significant.

    Parameters
    ----------
    counts : array-like
        The full count series; the LAST element is the point tested.
    baseline_years : int
        Number of previous years b contributing reference values.
    reference_window : int
        Half-width w of the calendar window around each anniversary.
    period : int
        Observations per year (52 for weekly data).
    alpha : float
        Two-sided level for the threshold.
    reweight : bool
        Apply the Anscombe-residual re-weighting pass.
    trend : bool
        Include the linear time trend.

    Returns
    -------
    result : dict
        Keys: estimate (exceedance score), observed, expected, threshold,
        alarm, phi, trend_coef, score, nbaseline, n, method.

    References
    ----------
    Farrington, Andrews, Beale & Catchpole (1996), JRSS-A 159(3):547-563,
    doi:10.2307/2983331.
    Noufaily, Enki, Farrington, Garthwaite, Andrews & Charlett (2013),
    Statistics in Medicine 32(7):1206-1222, doi:10.1002/sim.5595.
    """
    y = [float(v) for v in counts]
    n = len(y)
    if n == 0:
        raise ValueError("empty input: counts has no observations")
    if any(v < 0.0 for v in y):
        raise ValueError("counts must be non-negative")
    b = int(baseline_years)
    w = int(reference_window)
    per = int(period)
    if b < 1:
        raise ValueError("baseline_years must be positive")
    if w < 0:
        raise ValueError("reference_window must be non-negative")
    if per < 1:
        raise ValueError("period must be positive")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    t0 = n - 1
    idx = []
    for j in range(1, b + 1):
        c = t0 - j * per
        for d in range(-w, w + 1):
            k = c + d
            if 0 <= k < t0:
                idx.append(k)
    idx = sorted(set(idx))
    nb = len(idx)
    p = 2 if trend else 1
    if nb < p + 1:
        raise ValueError("not enough baseline observations (%d)" % nb)
    X = [([1.0, float(k - t0)] if trend else [1.0]) for k in idx]
    yb = [y[k] for k in idx]
    om = [1.0] * nb

    def _fit(om):
        beta, mu, inv = _irls_qpois(X, yb, om)
        dof = nb - p
        phi = sum(om[i] * (yb[i] - mu[i]) ** 2 / mu[i] for i in range(nb)) / dof
        if phi < 1.0:
            phi = 1.0
        return beta, mu, inv, phi

    beta, mu, inv, phi = _fit(om)
    if reweight:
        hat = []
        for i in range(nb):
            q = sum(X[i][r] * sum(inv[c][r] * X[i][c] for c in range(p))
                    for r in range(p))
            hat.append(om[i] * mu[i] * q)
        s = []
        for i in range(nb):
            an = 1.5 * (yb[i] ** (2.0 / 3.0) * mu[i] ** (-1.0 / 6.0)
                        - mu[i] ** 0.5)
            den = phi * (1.0 - hat[i])
            s.append(an / math.sqrt(den) if den > 0.0 else 0.0)
        # Farrington weights: gamma * s^-2 above 1, gamma below
        den = sum((v ** -2.0) if v > 1.0 else 1.0 for v in s)
        gam = nb / den if den > 0.0 else 1.0
        om = [gam * (v ** -2.0) if v > 1.0 else gam for v in s]
        beta, mu, inv, phi = _fit(om)
    x0 = [1.0, 0.0] if trend else [1.0]
    mu0 = math.exp(sum(x0[j] * beta[j] for j in range(p)))
    q0 = sum(x0[r] * sum(inv[c][r] * x0[c] for c in range(p)) for r in range(p))
    se0 = mu0 * math.sqrt(q0 if q0 > 0.0 else 0.0)
    tau = phi + se0 * se0 / mu0
    se = math.sqrt(4.0 / 9.0 * mu0 ** (1.0 / 3.0) * tau)
    z = core.qnorm(1.0 - a / 2.0)
    U = (mu0 ** (2.0 / 3.0) + z * se) ** 1.5
    y0 = y[t0]
    score = (y0 - mu0) / (U - mu0) if U > mu0 else float("inf")
    return RichResult(payload={
        "estimate": score,
        "observed": y0,
        "expected": mu0,
        "threshold": U,
        "alarm": 1.0 if y0 > U else 0.0,
        "phi": phi,
        "trend_coef": beta[1] if trend else 0.0,
        "score": score,
        "nbaseline": nb,
        "n": n,
        "method": "Farrington flexible algorithm",
    })


def cheatsheet():
    return "farsig: Farrington flexible algorithm"


# compact alias per ledger/NAMING.md
farringtonsignal = farrington_signal
