# morie.fn -- function file (rootcoder007/morie)
"""TMLE for treatment effects within covariate strata."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate as _tmle_ate

__all__ = ["tmle_heterogeneous"]


def tmle_heterogeneous(y, treatment, W, strata, trunc=0.01, min_stratum=20):
    r"""A separately targeted effect in each stratum, and a test that they differ.

    Fitting one TMLE per stratum is not the same as fitting one TMLE
    and slicing it. Each stratum gets its own targeting step, so the
    efficient influence function is solved WITHIN the stratum, and the
    resulting estimates are independent across strata -- which is what
    makes the heterogeneity test below legitimate.

    The test is the point. A set of stratum effects that look
    different is not evidence that they are: with :math:`K` strata and
    ordinary noise, the largest and smallest will always be some
    distance apart. Under the null of a constant effect

    .. math:: Q = \sum_k \frac{(\hat\tau_k - \bar\tau)^2}
              {\widehat{se}_k^2}
              \;\sim\; \chi^2_{K-1},

    with :math:`\bar\tau` the precision-weighted mean, and the p-value
    is reported alongside the range so that "these groups differ" has
    to survive a number.

    Strata that are too small to estimate, or that contain only one
    treatment arm, are dropped with a reason rather than returned as
    ``nan``: a stratum with no controls has no effect to estimate, and
    silently reporting one is how subgroup analyses go wrong.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome, binary or bounded continuous.
    treatment : array-like of {0, 1}, shape (n,)
        Treatment.
    W : array-like, shape (n, p) or (n,)
        Covariates used inside each stratum's nuisance fits.
    strata : array-like, shape (n,)
        Stratum labels. Any hashable type.
    trunc : float
        Propensity truncation within each stratum.
    min_stratum : int
        Strata smaller than this are dropped.

    Returns
    -------
    RichResult
        ``estimate`` (precision-weighted overall), ``se``, ``ci``,
        ``by_stratum`` (dict of label -> estimate/se/ci/n),
        ``heterogeneity_q``, ``heterogeneity_df``,
        ``heterogeneity_p``, ``range``, ``dropped``, ``n_strata``.

    References
    ----------
    van der Laan and Rose (2011), *Targeted Learning*, chapter 4.
    Cochran (1954), *Biometrics* 10:101-129 (the Q statistic).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> s = rng.integers(0, 2, size=n)
    >>> W = rng.normal(size=(n, 1))
    >>> A = (rng.uniform(size=n) < 0.5).astype(float)
    >>> y = 0.3 + 0.1 * W[:, 0] + A * (0.1 + 0.3 * s) + \
    ...     rng.normal(scale=0.1, size=n)
    >>> out = tmle_heterogeneous(y, A, W, s)
    >>> bool(out["heterogeneity_p"] < 0.01)
    True
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment, dtype=float).ravel()
    Wm = np.asarray(W, dtype=float)
    if Wm.ndim == 1:
        Wm = Wm[:, None]
    s = np.asarray(strata).ravel()
    n = y.size
    if not (A.size == n == Wm.shape[0] == s.size):
        raise ValueError(
            "y, treatment, W and strata must agree in length, got %d, %d, %d "
            "and %d." % (n, A.size, Wm.shape[0], s.size)
        )
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    labels = list(dict.fromkeys(s.tolist()))
    if len(labels) < 2:
        raise ValueError(
            "need at least 2 strata to speak of heterogeneity, got %d."
            % len(labels)
        )

    by, dropped = {}, {}
    for lab in labels:
        m = s == lab
        k = int(m.sum())
        if k < int(min_stratum):
            dropped[lab] = "only %d observations (min_stratum=%d)" % (
                k, int(min_stratum)
            )
            continue
        if A[m].sum() < 2 or (1 - A[m]).sum() < 2:
            dropped[lab] = (
                "%d treated and %d control: a stratum with one arm has no "
                "effect to estimate"
                % (int(A[m].sum()), int((1 - A[m]).sum()))
            )
            continue
        try:
            fit = _tmle_ate(y[m], A[m], Wm[m], trunc=trunc)
        except Exception as exc:                      # pragma: no cover
            dropped[lab] = "estimation failed: %s" % exc
            continue
        by[lab] = {
            "estimate": float(fit["ate"]),
            "se": float(fit["se"]),
            "ci": tuple(fit["ci"]),
            "n": k,
            "n_treated": int(A[m].sum()),
        }

    if len(by) < 2:
        raise ValueError(
            "only %d stratum survived the size and overlap checks; "
            "heterogeneity needs at least 2. Dropped: %s"
            % (len(by), dropped)
        )

    est = np.array([v["estimate"] for v in by.values()])
    se = np.array([v["se"] for v in by.values()])
    w = 1.0 / np.maximum(se**2, 1e-300)
    pooled = float(np.sum(w * est) / np.sum(w))
    pooled_se = float(np.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * (est - pooled) ** 2))
    df = int(est.size - 1)
    # upper-tail chi-square without a special-function dependency:
    # Wilson-Hilferty, exact enough for a reported p-value and
    # monotone in Q, which is what the decision needs
    z = ((q / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / np.sqrt(
        2.0 / (9 * df)
    )
    p = float(0.5 * math.erfc(z / np.sqrt(2.0)))

    zc = 1.959963984540054
    return RichResult(
        payload={
            "estimate": pooled,
            "se": pooled_se,
            "ci": (pooled - zc * pooled_se, pooled + zc * pooled_se),
            "by_stratum": by,
            "heterogeneity_q": q,
            "heterogeneity_df": df,
            "heterogeneity_p": p,
            "heterogeneity_note": (
                "stratum effects that look different are not evidence that "
                "they are; Q compares the spread against the estimates' own "
                "standard errors"
            ),
            "range": (float(est.min()), float(est.max())),
            "i_squared": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
            "dropped": dropped,
            "n_strata": len(by),
            "n": int(n),
            "per_stratum_targeting": (
                "each stratum is targeted separately, so its influence "
                "function is solved within the stratum and the estimates are "
                "independent across strata"
            ),
            "method": "Stratum-wise TMLE with a chi-square heterogeneity test",
        }
    )


def cheatsheet():
    return (
        "tmlhte: one targeted estimate per stratum plus a Q test that the "
        "strata actually differ; small or single-arm strata are dropped"
    )
