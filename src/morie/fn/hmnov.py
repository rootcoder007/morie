# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Novelty detection against a clean training distribution."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_novelty_detection"]


def _gaussian_model(train):
    """Fit a Gaussian to clean training data and return its log density."""
    mu = train.mean(axis=0)
    Xc = train - mu
    d = train.shape[1]
    S = (Xc.T @ Xc) / max(train.shape[0] - 1, 1) + 1e-9 * np.eye(d)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("geron_novelty_detection: the fitted covariance is singular; the training data are degenerate")
    Sinv = np.linalg.inv(S)

    def log_density(A, _mu=mu, _Si=Sinv, _ld=logdet, _d=d):
        B = np.atleast_2d(np.asarray(A, dtype=float))
        z = B - _mu
        m = np.einsum("ij,jk,ik->i", z, _Si, z)
        return -0.5 * (m + _ld + _d * np.log(2.0 * np.pi))

    return log_density


def geron_novelty_detection(model, X_new, reference=None):
    """
    Novelty detection: classify new points as novel vs in-distribution.

    Formula: novelty if p(x) / p_train < 1

    Novelty detection and anomaly detection run the same test on
    different assumptions: here the training set is assumed CLEAN, so
    the density is a description of normality rather than a compromise
    with the outliers already in it. That assumption is the whole
    difference, and it is why a novelty detector trained on contaminated
    data quietly stops working.

    The reference density p_train is the geometric mean of the training
    densities, exp(mean log p) -- a typical-set level rather than a peak,
    so a point is novel when it is less likely than a typical training
    point, not merely less likely than the mode.

    ``model`` may be a callable log density, a mapping with a
    ``log_density`` callable (and optionally a ``reference``), or the
    clean training array itself, in which case a Gaussian is fitted.

    Parameters
    ----------
    model : callable, mapping or array-like
        As described.
    X_new : array-like, shape (m, d)
        Points to test.
    reference : float, optional
        Log reference density, overriding the fitted one.

    Returns
    -------
    result : RichResult
        Keys: ratio, log_ratio, is_novel, log_density, reference,
        novel_fraction, estimate, n, method.

    Examples
    --------
    A Gaussian fitted to a tight cluster flags a distant point:

    >>> train = [[0.0], [0.1], [-0.1], [0.05], [-0.05]]
    >>> r = geron_novelty_detection(train, [[0.0], [10.0]])
    >>> [bool(v) for v in r["is_novel"]]
    [False, True]
    >>> bool(r["ratio"][0] > 1.0 > r["ratio"][1])
    True
    >>> float(r["novel_fraction"])
    0.5

    A callable log density is used directly:

    >>> f = lambda A: -np.abs(np.asarray(A, dtype=float)).ravel()
    >>> [bool(v) for v in geron_novelty_detection(f, [[0.0], [5.0]], reference=-1.0)["is_novel"]]
    [False, True]

    References
    ----------
    Geron Ch 8
    """
    B = np.asarray(X_new, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.size == 0:
        raise ValueError(f"geron_novelty_detection: X_new must be a non-empty 2-D array, got shape {B.shape}")
    if not np.all(np.isfinite(B)):
        raise ValueError("geron_novelty_detection: X_new contains non-finite values")

    ref = None if reference is None else float(reference)
    if callable(model):
        log_density = model
    elif hasattr(model, "get") and not isinstance(model, (list, tuple, np.ndarray)):
        log_density = model.get("log_density")
        if not callable(log_density):
            raise ValueError("geron_novelty_detection: the model mapping needs a callable 'log_density'")
        if ref is None and model.get("reference") is not None:
            ref = float(model["reference"])
    else:
        train = np.asarray(model, dtype=float)
        if train.ndim == 1:
            train = train.reshape(-1, 1)
        if train.ndim != 2 or train.shape[0] < 2:
            raise ValueError(f"geron_novelty_detection: training data must be 2-D with at least 2 rows, got shape {train.shape}")
        if train.shape[1] != B.shape[1]:
            raise ValueError(f"geron_novelty_detection: training data has {train.shape[1]} features but X_new has {B.shape[1]}")
        log_density = _gaussian_model(train)
        if ref is None:
            ref = float(np.mean(log_density(train)))
    if ref is None:
        raise ValueError(
            "geron_novelty_detection: no reference density; pass reference= or give the clean training data as model"
        )

    ld = np.asarray(log_density(B), dtype=float).ravel()
    if ld.size != B.shape[0]:
        raise ValueError(f"geron_novelty_detection: the model returned {ld.size} densities for {B.shape[0]} rows")
    if not np.all(np.isfinite(ld)):
        raise ValueError("geron_novelty_detection: the model returned non-finite log densities")

    log_ratio = ld - ref
    ratio = np.exp(np.clip(log_ratio, -700, 700))
    novel = log_ratio < 0
    return RichResult(
        title="Novelty detection",
        summary_lines=[("Points", int(B.shape[0])), ("Novel fraction", float(np.mean(novel))), ("Reference log density", ref)],
        interpretation="Assumes the training set was clean; contamination silently rescales the reference.",
        payload={
            "ratio": ratio,
            "log_ratio": log_ratio,
            "is_novel": novel,
            "log_density": ld,
            "reference": ref,
            "novel_fraction": float(np.mean(novel)),
            "estimate": ratio,
            "n": int(B.shape[0]),
            "method": "Density-ratio novelty test against a typical-set reference",
        },
    )


def cheatsheet():
    return "hmnov: Novelty detection by density ratio against clean training data"
