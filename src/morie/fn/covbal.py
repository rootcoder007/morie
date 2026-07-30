# morie.fn -- function file (rootcoder007/morie)
"""Covariate balance check."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["covariate_balance_check"]


def covariate_balance_check(X, treat, weights=None, threshold=0.1):
    r"""Standardised mean differences before and after weighting.

    .. math::
        \mathrm{SMD}_j = \frac{\bar x_{1j} - \bar x_{0j}}
                              {\sqrt{(s_{1j}^2 + s_{0j}^2)/2}},

    with the pooled denominator computed on the **unweighted** sample in both
    cases, so that before and after are on the same scale and the comparison
    means something.

    The standardised difference is used rather than a t-test for a specific
    reason: a hypothesis test conflates imbalance with sample size, so in a
    large study a trivial difference is "significant" and in a small one a
    serious imbalance is not. Balance is a property of the sample in hand, not
    a population claim, so a significance test is the wrong instrument
    entirely -- which is why the 0.1 threshold on the SMD is the convention.

    Balance on means is necessary and not sufficient. Two groups can match on
    every mean while differing in variance or in the shape of the tails, so
    ``variance_ratio`` is reported alongside; it should sit near 1, and
    anything outside roughly [0.5, 2] is a problem the SMD cannot see.

    Parameters
    ----------
    X : array-like
        Covariates ``(n, p)``.
    treat : array-like
        Treatment indicator, 0/1.
    weights : array-like, optional
        Balancing weights. Without them only the raw balance is reported.
    threshold : float
        SMD threshold for flagging imbalance.

    Returns
    -------
    RichResult
        ``smd_before``, ``smd_after``, ``variance_ratio``, ``n_imbalanced``,
        ``balanced``, ``worst``.

    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the distribution
        of baseline covariates between treatment groups in propensity-score
        matched samples. *Statistics in Medicine*, 28(25), 3083-3107.
    Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
        Social, and Biomedical Sciences*. Cambridge University Press.

    Examples
    --------
    Weighting on a correct propensity model improves balance.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(2000, 3))
    >>> ps = 1 / (1 + np.exp(-(0.8 * X[:, 0] + 0.5 * X[:, 1])))
    >>> tr = (rng.random(2000) < ps).astype(float)
    >>> w = np.where(tr == 1, 1 / ps, 1 / (1 - ps))
    >>> r = covariate_balance_check(X, tr, weights=w)
    >>> bool(np.max(np.abs(r["smd_after"])) < np.max(np.abs(r["smd_before"])))
    True

    The confounded covariates are flagged before weighting.

    >>> bool(abs(r["smd_before"][0]) > 0.1)
    True

    Variance ratios are reported because means can match while shapes do not.

    >>> bool(np.all(np.isfinite(r["variance_ratio"])))
    True

    >>> covariate_balance_check(X, tr[:10])
    Traceback (most recent call last):
        ...
    ValueError: X has 2000 rows but treat has 10
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if X.shape[0] != tr.size:
        raise ValueError(f"X has {X.shape[0]} rows but treat has {tr.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    t1, t0 = tr == 1, tr == 0
    if not t1.any() or not t0.any():
        raise ValueError("both treatment groups must be non-empty")

    # Pooled sd from the UNWEIGHTED sample in both cases, so before and after
    # are measured on the same scale.
    v1 = X[t1].var(axis=0, ddof=1)
    v0 = X[t0].var(axis=0, ddof=1)
    pooled = np.sqrt(np.maximum((v1 + v0) / 2.0, 1e-300))
    smd_before = (X[t1].mean(axis=0) - X[t0].mean(axis=0)) / pooled

    if weights is None:
        smd_after = smd_before
        vr = v1 / np.maximum(v0, 1e-300)
    else:
        w = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
        if w.size != tr.size:
            raise ValueError(f"weights has {w.size} entries but treat has {tr.size}")
        def wmean(m):
            ww = w[m]
            return (X[m] * ww[:, None]).sum(axis=0) / max(ww.sum(), 1e-300)
        def wvar(m):
            ww = w[m]
            mu = wmean(m)
            s = ww.sum()
            return ((X[m] - mu) ** 2 * ww[:, None]).sum(axis=0) / max(s, 1e-300)
        smd_after = (wmean(t1) - wmean(t0)) / pooled
        vr = wvar(t1) / np.maximum(wvar(t0), 1e-300)

    bad = int(np.sum(np.abs(smd_after) > threshold))
    worst = int(np.argmax(np.abs(smd_after)))
    return RichResult(
        title="Covariate balance",
        summary_lines=[("p", int(X.shape[1])), ("imbalanced", bad),
                       ("max |SMD|", float(np.max(np.abs(smd_after))))],
        warnings=(["balance on means is necessary, not sufficient; check "
                   "variance_ratio, which should sit near 1"]
                  + ([f"{bad} covariates exceed |SMD| = {threshold}"] if bad else [])),
        payload={
            "smd_before": smd_before, "smd_after": smd_after,
            "variance_ratio": vr, "n_imbalanced": bad,
            "balanced": bool(bad == 0), "worst": worst,
            "threshold": float(threshold), "method": "covariate_balance_check",
        },
    )


def cheatsheet():
    return "covbal: SMD not a t-test (tests conflate imbalance with n); means matching is necessary, not sufficient"
