# morie.fn -- function file (rootcoder007/morie)
r"""The forward search: fit the clean data first, watch what breaks.

**The masking problem.** Deletion diagnostics ask what happens when
one observation is removed. With two outliers that reinforce each
other, removing either changes almost nothing -- each masks the
other -- and the diagnostic reports a clean data set. Robust
estimators fix this by downweighting, but they hand back one fit and
no account of which observations were fought over.

**The idea.** Order the data by how well it agrees with itself. Start
from a small subset chosen to be outlier-free, fit it, and rank every
observation by squared residual. Take the :math:`m+1` best as the
next subset, refit, re-rank. Repeat to the full sample.

Outliers cannot mask each other here, because they are not in the
subset while the model is being estimated. They enter at the very
end, and when they do the fitted quantities jump.

**Reading the search.** What matters is not the final fit -- that is
just least squares on everything -- but the *trajectory*:

* the minimum deletion residual among observations outside the
  subset spikes at the step where the first outlier is forced in;
* the estimate of :math:`\sigma` *steps* when contamination enters.
  It does not sit flat before that: subsets chosen to fit well are
  biased low, so :math:`s` drifts upward through the search even on
  perfectly clean data as that selection bias unwinds. The signature
  of an outlier is a jump between consecutive steps, not the overall
  range;
* a coefficient that moves sharply at one step is being driven by
  whatever entered at that step.

The step at which the jump happens says *how many* outliers there
are, which is exactly what a single robust fit does not tell you.

**Choosing the start.** Least median of squares over random
:math:`p`-subsets: draw subsets, fit each exactly, keep the one with
the smallest median squared residual. Deterministic given the seed,
and reported.

**The first steps are not evidence.** At :math:`m` barely above
:math:`p` the subset fits almost exactly, :math:`s` is near zero, and
a deletion residual divided by it is enormous whatever the data looks
like. Threshold rules that ignore this fire on perfectly clean
samples. Monitoring therefore starts only once the subset carries
enough residual degrees of freedom to estimate a scale at all --
``min_df``, defaulting to five. The full trajectory is still
returned; it is the *rule* that waits, not the search.

**The honest limit.** The search orders observations; it does not
label them. A large spike says the model changed, not that the point
is an error. Reading a spike as "delete this row" is how a real
observation gets thrown away for the crime of being informative.

References
----------
Atkinson, A. C. & Riani, M. (2000) *Robust Diagnostic Regression
Analysis*, Springer Series in Statistics, ISBN 978-1-4612-7027-0,
doi:10.1007/978-1-4612-1160-0. The forward search: the least-median-
of-squares start, the residual ordering that defines each subset, the
monitoring of deletion residuals and of :math:`s^2` along the search,
and the reading of the step at which quantities jump as the number of
outliers.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ols_fit", "lms_start", "forward_search", "forward_plot",
           "forward_search_regression", "consistency_factor"]


def _prep(X, y):
    M = [[float(v) for v in row] for row in X]
    yy = [float(v) for v in y]
    n = len(M)
    if n != len(yy):
        raise ValueError("forwsr: %d rows of X but %d responses"
                         % (n, len(yy)))
    if n < 4:
        raise ValueError("forwsr: need at least four observations")
    p = len(M[0]) if M else 0
    if p == 0 or any(len(r) != p for r in M):
        raise ValueError("forwsr: the design is ragged or empty")
    if n <= p:
        raise ValueError("forwsr: %d observations cannot support %d "
                         "coefficients" % (n, p))
    return M, yy, n, p


def _solve(A, b):
    p = len(b)
    Ab = [list(A[i]) + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(Ab[r][c]))
        if abs(Ab[piv][c]) < 1e-12:
            raise ValueError("forwsr: the subset is rank deficient; "
                             "its design has collinear columns")
        Ab[c], Ab[piv] = Ab[piv], Ab[c]
        for r in range(p):
            if r == c:
                continue
            f = Ab[r][c] / Ab[c][c]
            for k in range(c, p + 1):
                Ab[r][k] -= f * Ab[c][k]
    return [Ab[i][p] / Ab[i][i] for i in range(p)]


def _norm_ppf(p):
    """Standard normal quantile by bisection on the error function."""
    if not 0.0 < p < 1.0:
        raise ValueError("forwsr: a probability must lie in (0, 1), "
                         "got %r" % (p,))
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def consistency_factor(m, n):
    r"""Riani, Atkinson & Cerioli's factor for a truncated scale.

    The subset of size :math:`m` is the :math:`m` observations with the
    SMALLEST squared residuals, so :math:`s^2_m` estimates the variance
    of a truncated normal, not of the normal. It is biased low, and the
    bias unwinds as :math:`m 	o n`: on clean data the raw scale still
    climbs by a factor of sixteen across the search. Comparing a
    deletion residual against a fixed threshold without this correction
    therefore flags clean data at almost every step.

    Their equation (12) gives the inflation factor for :math:`v`
    dimensions as :math:`c_{FS}(m) = (m/n) / P(\chi^2_{v+2} <
    \chi^2_{v,m/n})`. For regression residuals :math:`v = 1`, and with
    :math:`\psi = \Phi^{-1}\{(n+m)/2n\}` the denominator is
    :math:`2\Phi(\psi) - 1 - 2\psi\phi(\psi) = m/n - 2\psi\phi(\psi)`,
    so the factor reduces exactly to

    .. math:: 1/c_{FS}(m) = 1 - (2n/m)\,\psi\,\phi(\psi).

    Returns that reciprocal: multiply :math:`s^2_m` by it to get a
    consistent estimate of :math:`\sigma^2`.

    References
    ----------
    Riani, M., Atkinson, A. C. and Cerioli, A. (2009) "Finding an
    unknown number of multivariate outliers", Journal of the Royal
    Statistical Society Series B 71(2), 447-466,
    doi:10.1111/j.1467-9868.2008.00692.x, equation (12) and section
    4.2. Fetched and read.
    """
    m, n = int(m), int(n)
    if m >= n:
        return 1.0
    if m <= 0:
        raise ValueError("forwsr: the subset cannot be empty")
    psi = _norm_ppf((n + m) / (2.0 * n))
    phi = math.exp(-0.5 * psi * psi) / math.sqrt(2.0 * math.pi)
    c = 1.0 - (2.0 * n / m) * psi * phi
    return c if c > 0.0 else 1.0


def ols_fit(X, y, subset=None):
    r"""Least squares on ``subset`` (all rows if omitted)."""
    M, yy, n, p = _prep(X, y)
    idx = list(range(n)) if subset is None else [int(i) for i in subset]
    if len(idx) < p:
        raise ValueError("forwsr: a subset of %d cannot fit %d "
                         "coefficients" % (len(idx), p))
    A = [[sum(M[i][a] * M[i][b] for i in idx) for b in range(p)]
         for a in range(p)]
    v = [sum(M[i][a] * yy[i] for i in idx) for a in range(p)]
    beta = _solve(A, v)
    resid = [yy[i] - sum(M[i][k] * beta[k] for k in range(p))
             for i in range(n)]
    df = len(idx) - p
    s2 = (sum(resid[i] ** 2 for i in idx) / df) if df > 0 else 0.0
    return {"beta": beta, "residuals": resid, "s2": s2,
            "sigma": math.sqrt(s2), "subset": idx, "df": df}


def lms_start(X, y, n_draw=500, seed=1):
    r"""Least median of squares over random p-subsets.

    Returns the p-subset whose fit has the smallest median squared
    residual -- a starting point unlikely to contain an outlier.
    """
    M, yy, n, p = _prep(X, y)
    rng = np.random.default_rng(int(seed))
    best, best_med = None, float("inf")
    for _ in range(int(n_draw)):
        idx = []
        for _k in range(p):
            j = int(rng.random() * n) % n
            while j in idx:
                j = int(rng.random() * n) % n
            idx.append(j)
        try:
            f = ols_fit(X, y, idx)
        except ValueError:
            continue
        sq = sorted(r * r for r in f["residuals"])
        med = sq[len(sq) // 2]
        if med < best_med:
            best_med, best = med, sorted(idx)
    if best is None:
        raise ValueError("forwsr: every sampled subset was rank "
                         "deficient; is the design collinear?")
    return {"subset": best, "median_sq_residual": best_med}


def forward_search(X, y, start=None, n_draw=500, seed=1):
    r"""Run the search from ``m = p`` to ``m = n``, monitoring as it
    goes."""
    M, yy, n, p = _prep(X, y)
    if start is None:
        cur = list(lms_start(X, y, n_draw, seed)["subset"])
    else:
        cur = sorted(int(i) for i in start)
        if len(cur) < p:
            raise ValueError("forwsr: the starting subset must hold "
                             "at least %d observations" % p)
    steps = []
    while True:
        f = ols_fit(X, y, cur)
        outside = [i for i in range(n) if i not in cur]
        # The minimum deletion residual among observations NOT in the
        # subset: the quantity that spikes when an outlier is forced
        # in. Scaled by the subset's own sigma.
        if outside and f["sigma"] > 0:
            # scale the truncated s before comparing anything to it
            sig = f["sigma"] / math.sqrt(consistency_factor(len(cur), n))
            mdr = (min(abs(f["residuals"][i]) for i in outside) / sig
                   if sig > 0.0 else float("nan"))
        else:
            mdr = float("nan")
        cfac = consistency_factor(len(cur), n)
        steps.append({"m": len(cur), "beta": list(f["beta"]),
                      "sigma": f["sigma"], "s2": f["s2"],
                      "consistency_factor": cfac,
                      "sigma_corrected": f["sigma"] / math.sqrt(cfac),
                      "min_deletion_residual": mdr,
                      "subset": list(cur)})
        if len(cur) >= n:
            break
        order = sorted(range(n), key=lambda i: abs(f["residuals"][i]))
        cur = sorted(order[:len(cur) + 1])
    return steps


def forward_plot(steps, key="min_deletion_residual"):
    r"""The monitored series along the search, for reading the jump."""
    if not steps:
        raise ValueError("forwsr: no steps to monitor")
    if key not in steps[0]:
        raise ValueError("forwsr: %r is not monitored; available: %s"
                         % (key, ", ".join(sorted(k for k in steps[0]
                                                  if k != "subset"))))
    return {"m": [s["m"] for s in steps],
            key: [s[key] for s in steps]}


def forward_search_regression(X, y, start=None, n_draw=500, seed=1,
                              threshold=3.0, min_df=5):
    r"""Entry point: run the search and report where it jumps.

    ``threshold`` is on the minimum deletion residual. The units
    flagged are those entering after the first exceedance --
    candidates to look at, not verdicts.

    ``min_df`` holds the rule back until the subset has that many
    residual degrees of freedom. Without it the rule fires on the
    opening steps of any data set, clean or not, because dividing by
    a near-zero scale gives a near-infinite ratio.
    """
    M, _yy, _n0, p = _prep(X, y)
    steps = forward_search(X, y, start, n_draw, seed)
    n = steps[-1]["m"]
    jump = None
    for s in steps:
        if s["m"] - p < int(min_df):
            continue
        v = s["min_deletion_residual"]
        if v == v and v > float(threshold):
            jump = s["m"]
            break
    entered = []
    for a, b in zip(steps, steps[1:]):
        new = [i for i in b["subset"] if i not in a["subset"]]
        entered.append((b["m"], new[0] if new else None))
    flagged = [i for m, i in entered
               if jump is not None and m > jump and i is not None]
    full = ols_fit(X, y)
    return RichResult(payload={
        "estimate": full["beta"], "coefficients": full["beta"],
        "steps": steps, "n": n, "n_flagged": len(flagged),
        "flagged": flagged, "jump_at_m": jump,
        "threshold": float(threshold), "min_df": int(min_df),
        "monitored_from_m": p + int(min_df),
        "entry_order": entered,
        "sigma_trajectory": [s["sigma"] for s in steps],
        "mdr_trajectory": [s["min_deletion_residual"] for s in steps],
        "method": "forward search (Atkinson & Riani 2000) from a "
                  "least-median-of-squares start",
    })
