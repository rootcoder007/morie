# morie.fn -- function file (rootcoder007/morie)
"""Mahalanobis-distance matching."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_mahalanobis_match"]


def causal_mahalanobis_match(X, treat, k=1, replace=True, caliper=None):
    r"""Match treated to control units on Mahalanobis distance.

    .. math::
        d(i, j) = \sqrt{(x_i - x_j)^\top S^{-1} (x_i - x_j)},

    with :math:`S` the pooled covariance. The inverse covariance is what makes
    this different from Euclidean matching: it rescales each covariate by its
    spread **and** removes the double-counting of correlated covariates. Two
    covariates correlated at 0.9 count as roughly one dimension, not two.

    Mahalanobis matching works well in low dimension and degrades badly as
    dimension grows -- with many covariates all pairwise distances concentrate,
    so the "nearest" control is barely nearer than an arbitrary one. Beyond
    roughly eight covariates, propensity-score matching is usually the better
    tool, and that boundary is flagged.

    ``replace=True`` reuses controls, which lowers bias and raises variance
    because the effective control sample shrinks; ``reuse_max`` reports how
    hard a single control is working.

    Parameters
    ----------
    X : array-like
        Covariates ``(n, p)``.
    treat : array-like
        Treatment indicator, 0/1.
    k : int
        Controls matched per treated unit.
    replace : bool
        Allow a control to serve several treated units.
    caliper : float, optional
        Maximum acceptable distance, in Mahalanobis units.

    Returns
    -------
    RichResult
        ``matches``, ``distances``, ``matched_treated``, ``n_unmatched``,
        ``reuse_max``, ``mean_distance``.

    References
    ----------
    Rubin, D. B. (1980). Bias reduction using Mahalanobis-metric matching.
        *Biometrics*, 36(2), 293-298.
    Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
        Social, and Biomedical Sciences*. Cambridge University Press.

    Examples
    --------
    Matching finds close controls, so the mean matched distance is far below
    the mean distance to a random control.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> tr = (rng.random(400) < 0.3).astype(float)
    >>> r = causal_mahalanobis_match(X, tr, k=1)
    >>> bool(r["mean_distance"] < 0.5)
    True

    Correlated covariates are not double-counted: the metric is invariant to
    an invertible linear transform of the covariates, so rescaling one column
    leaves the matches unchanged.

    >>> X2 = X.copy(); X2[:, 0] *= 100
    >>> a = causal_mahalanobis_match(X, tr, k=1)["matches"]
    >>> b = causal_mahalanobis_match(X2, tr, k=1)["matches"]
    >>> bool(np.array_equal(a, b))
    True

    A caliper leaves poor matches unmatched rather than accepting them.

    >>> tight = causal_mahalanobis_match(X, tr, k=1, caliper=0.05)
    >>> bool(tight["n_unmatched"] > 0)
    True

    High dimension is flagged, since distances concentrate there.

    >>> Xw = rng.normal(size=(200, 12))
    >>> bool(causal_mahalanobis_match(Xw, (rng.random(200) < 0.3).astype(float)).warnings)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if X.shape[0] != tr.size:
        raise ValueError(f"X has {X.shape[0]} rows but treat has {tr.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    ti = np.flatnonzero(tr == 1)
    ci = np.flatnonzero(tr == 0)
    if ti.size == 0 or ci.size == 0:
        raise ValueError("both treatment groups must be non-empty")
    if not replace and ci.size < k * ti.size:
        raise ValueError(
            f"matching without replacement needs {k * ti.size} controls but "
            f"only {ci.size} are available"
        )

    S = np.cov(X, rowvar=False).reshape(X.shape[1], X.shape[1])
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    D = np.empty((ti.size, ci.size))
    for a, i in enumerate(ti):
        d = X[ci] - X[i]
        D[a] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Sinv, d), 0.0))

    matches = np.full((ti.size, k), -1, dtype=int)
    dists = np.full((ti.size, k), np.nan)
    used = np.zeros(ci.size, dtype=int)
    for a in range(ti.size):
        order = np.argsort(D[a])
        picked = 0
        for j in order:
            if not replace and used[j] > 0:
                continue
            if caliper is not None and D[a, j] > caliper:
                break
            matches[a, picked] = int(ci[j])
            dists[a, picked] = D[a, j]
            used[j] += 1
            picked += 1
            if picked == k:
                break

    ok = matches[:, 0] >= 0
    warn = []
    if X.shape[1] > 8:
        warn.append(
            f"{X.shape[1]} covariates: Mahalanobis distances concentrate in high "
            "dimension, so the nearest control is barely nearer than an "
            "arbitrary one; prefer propensity-score matching"
        )
    if replace and used.max() > max(3, ti.size // 10):
        warn.append(f"one control is matched to {int(used.max())} treated units; "
                    "the effective control sample is much smaller than it looks")
    return RichResult(
        title="Mahalanobis matching",
        summary_lines=[("treated", int(ti.size)), ("matched", int(ok.sum())),
                       ("mean distance", float(np.nanmean(dists[ok])) if ok.any() else np.nan)],
        warnings=warn,
        payload={
            "matches": matches, "distances": dists,
            "matched_treated": ti[ok], "n_unmatched": int((~ok).sum()),
            "reuse_max": int(used.max()) if used.size else 0,
            "mean_distance": float(np.nanmean(dists[ok])) if ok.any() else float("nan"),
            "treated_index": ti, "control_index": ci,
            "method": "causal_mahalanobis_match",
        },
    )


def cheatsheet():
    return "causmm: S^-1 makes it invariant to linear rescaling; distances concentrate past ~8 covariates"
