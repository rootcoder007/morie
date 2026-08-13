# morie.fn -- function file (rootcoder007/morie)
r"""Lagged-value IPTW.

Robins (1986) introduced the g-formula and inverse-probability weighting
for a *sustained exposure period*: treatment is repeated over time, and
the covariates that predict the next treatment are themselves affected
by the previous one. Section 6 of that paper is the origin of the
weighted estimator that Hernan & Robins later present as Sec. 21.2.

The specific thing this module implements is what makes the estimator
usable in practice: the treatment model at time k conditions on the
*lagged values* of the outcome and covariate processes, not just on
their current values. Writing :math:`\bar{A}_{k-1}` for treatment
history and :math:`\bar{L}_k` for covariate history, the denominator is

.. math:: f\!\left(A_k \mid \bar{A}_{k-1},\, L_k, L_{k-1}, \ldots,
                            L_{k-\ell},\, Y_{k-1}, \ldots, Y_{k-\ell}\right)

for a lag :math:`\ell`, and the weight is the product over k of the
Sec. 21.2 ratio. The stub this replaces said "include Y_{t-1} in
propensity model" and then returned ``mean(y)``.

**Why the lag matters and is not decoration.** The whole difficulty
Robins (1986) identified is that a time-varying confounder affected by
prior treatment cannot be handled by conditioning -- adjust for it and
you block part of the treatment effect, omit it and you leave
confounding. Weighting solves it, but only if the weight model captures
the dependence, and with a sustained exposure that dependence reaches
back more than one period. ``lag=0`` reduces to the contemporaneous
model, which is the common misuse; the anchor shows the two disagree.

References
----------
Robins, J. (1986) "A new approach to causal inference in mortality
studies with a sustained exposure period -- application to control of
the healthy worker survivor effect", *Mathematical Modelling* 7(9-12),
1393-1512, doi:10.1016/0270-0255(86)90088-6.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*, Boca
Raton: Chapman & Hall/CRC, Sec. 21.2 for the product-over-time weights
and Ch. 20 for why conditioning fails where weighting works.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["laggedval_iptw", "lagged_design"]


def lagged_design(L_hist, Y_hist=None, k_time=0, lag=1):
    """Covariates at time `k_time` together with `lag` earlier values.

    Missing history at the start of follow-up is not invented: a lag
    that reaches before time 0 simply contributes no columns, so early
    periods have a smaller model than late ones. Padding with zeros
    would tell the treatment model that the process was at zero before
    the study began, which is a claim about the data nobody made.
    """
    cols = []
    lo = max(0, int(k_time) - int(lag))
    for j in range(int(k_time), lo - 1, -1):
        if L_hist[j] is not None:
            block = k.mat(L_hist[j])
            for c in range(len(block[0])):
                cols.append([row[c] for row in block])
    if Y_hist is not None:
        for j in range(int(k_time) - 1, lo - 1, -1):
            if j >= 0 and Y_hist[j] is not None:
                cols.append([float(v) for v in k.vec(Y_hist[j])])
    return cols


def laggedval_iptw(y, A, H, lag=1, Y_hist=None, stabilize=True,
                   kind="binary", trim=None, contrast="cumulative"):
    r"""IPTW for a sustained exposure, with lagged values in the model.

    Parameters
    ----------
    y : array-like
        End-of-follow-up outcome, one value per subject.
    A : list of array-like
        Treatment at each time point, ``A[k][i]``. A single vector is
        accepted and treated as one time point.
    H : list of array-like
        Time-varying covariates, ``H[k]`` an n-by-p block. ``None``
        entries mean no measured covariates at that time.
    lag : int
        How many earlier periods of `H` (and of `Y_hist`, if given) to
        put in the treatment model alongside the current one.
    Y_hist : list of array-like, optional
        Intermediate outcomes, so that :math:`Y_{k-1}` enters the model
        for :math:`A_k` -- the "lagged value" the module is named for.
    contrast : {"cumulative", "final", "everexposed"}
        The marginal structural model fitted to the weighted data.
        "cumulative" regresses y on total exposure sum_k A_k, which is
        the dose-response MSM; "final" on the last treatment only;
        "everexposed" on the indicator that any A_k was 1.

    Returns
    -------
    RichResult
        ``estimate`` is the MSM coefficient on the exposure summary.

    Examples
    --------
    Two periods of treatment with a lagged confounder::

        r = laggedval_iptw(y, [A0, A1], [L0, L1], lag=1)
        r["estimate"]
    """
    if contrast not in ("cumulative", "final", "everexposed"):
        raise ValueError(
            "laggedval_iptw: contrast must be 'cumulative', 'final' or "
            "'everexposed', got %r" % (contrast,))
    lag = int(lag)
    if lag < 0:
        raise ValueError("laggedval_iptw: lag must be non-negative, got %r"
                         % (lag,))
    A_hist = _as_history(A)
    K = len(A_hist)
    L_hist = _as_history(H, allow_none=True)
    if len(L_hist) < K:
        L_hist = list(L_hist) + [None] * (K - len(L_hist))
    yv = k.vec(y)
    n = len(yv)
    for kk in range(K):
        if len(k.vec(A_hist[kk])) != n:
            raise ValueError(
                "laggedval_iptw: outcome has %d rows but treatment at "
                "time %d has %d" % (n, kk, len(k.vec(A_hist[kk]))))

    # Sec. 21.2's product, with the lagged design at each time point.
    w = [1.0] * n
    per_time = []
    past = []
    for kk in range(K):
        ak = k.vec(A_hist[kk])
        cols = lagged_design(L_hist, Y_hist, kk, lag)
        den_X = _bind(cols + past, n)
        num_X = _bind(past, n) if past else None
        wk, info = k.ip_weights(ak, den_X, num_X, kind=kind,
                                stabilize=stabilize)
        for i in range(n):
            w[i] *= wk[i]
        per_time.append({"time": kk, "n_covariates":
                         len(cols) + len(past), "info": info})
        past = past + [list(ak)]
    if trim is not None:
        q = float(trim)
        if not 0.5 < q < 1.0:
            raise ValueError("laggedval_iptw: trim must be in (0.5, 1)")
        hi = k.quantile7(w, q)
        lo = k.quantile7(w, 1.0 - q)
        w = [min(max(v, lo), hi) for v in w]

    cum = [sum(k.vec(A_hist[kk])[i] for kk in range(K)) for i in range(n)]
    if contrast == "cumulative":
        X = [[cum[i]] for i in range(n)]
    elif contrast == "final":
        last = k.vec(A_hist[-1])
        X = [[last[i]] for i in range(n)]
    else:
        X = [[1.0 if cum[i] > 0.0 else 0.0] for i in range(n)]
    fit = k.wls(X, yv, w)
    s1 = sum(w)
    s2 = sum(v * v for v in w)
    return RichResult(payload={
        "estimate": fit["coef"][1],
        "se": fit["se"][1],
        "intercept": fit["coef"][0],
        "coef": fit["coef"],
        "vcov": fit["vcov"],
        "weights": w,
        "mean_weight": s1 / n,
        "max_weight": max(w),
        "effective_sample_size": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
        "cumulative_exposure": cum,
        "per_time": per_time,
        "n_times": K, "lag": lag, "n": n, "contrast": contrast,
        "method": "lagged-value IPTW, Robins (1986); weights by "
                  "Hernan & Robins (2020) Sec. 21.2",
    })


def _as_history(obj, allow_none=False):
    if obj is None:
        return [None] if allow_none else []
    if isinstance(obj, (list, tuple)) and obj and (
            isinstance(obj[0], (list, tuple)) or obj[0] is None
            or hasattr(obj[0], "shape")):
        return list(obj)
    return [obj]


def _bind(cols, n):
    if not cols:
        return None
    return [[float(c[i]) for c in cols] for i in range(n)]


def cheatsheet():
    return ("lggvls: sustained-exposure IPTW (Robins 1986). Weight = "
            "prod_k f(A_k|Abar_{k-1}) / f(A_k|Abar_{k-1}, Lbar_k, "
            "L_{k-1..k-lag}, Y_{k-1..k-lag}); MSM on cumulative, final "
            "or ever-exposed contrast.")


# compact alias per ledger/NAMING.md
laggedvaliptw = laggedval_iptw
