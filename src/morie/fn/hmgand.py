# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GMM-based anomaly detection: low-density points are anomalies."""

import numpy as np

from ._richresult import RichResult
from .hmgmm import geron_gaussian_mixture, gmm_log_pdf

__all__ = ["geron_anomaly_gmm"]


def geron_anomaly_gmm(X, n_components=2, threshold=None, contamination=0.05, seed=0, X_new=None):
    """
    GMM-based anomaly detection: low-density points are anomalies.

    Formula: anomaly if p(x) < threshold

    The mixture is fitted by EM -- DELEGATED to
    :func:`morie.fn.hmgmm.geron_gaussian_mixture` -- and the density
    ``p(x) = sum_k pi_k N(x; mu_k, S_k)`` is then evaluated in log space,
    because in more than a couple of dimensions the raw density
    underflows to 0 for every point and every comparison becomes
    ``0 < 0``.

    ``threshold`` may be given directly as a density. Otherwise it is set
    to the ``contamination`` quantile of the training densities, which is
    the honest way to pick it: an absolute density has no scale-free
    meaning, so the cut is defined by the fraction of the training data
    you are willing to call anomalous.

    ``X_new`` scores unseen points against the same fitted mixture and
    threshold, which is how the detector is actually used.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Training data.
    n_components : int, default 2
    threshold : float, optional
        Absolute density cut; overrides ``contamination``.
    contamination : float, default 0.05
        Fraction of training points flagged, in (0, 1).
    seed : int, default 0
    X_new : array-like, optional
        Points to score with the fitted model.

    Returns
    -------
    result : RichResult
        Keys: is_anomaly, density, log_density, threshold, n_anomalies,
        anomaly_indices, new_density, new_is_anomaly, gmm, estimate, n,
        method.

    Examples
    --------
    Two tight clusters and one far-off point: at 20% contamination the
    outlier is exactly the flagged one.

    >>> X = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2], [100.0], [0.05], [10.05], [0.15]]
    >>> r = geron_anomaly_gmm(X, n_components=2, contamination=0.1, seed=1)
    >>> r["anomaly_indices"]
    [6]
    >>> r["n_anomalies"]
    1
    >>> r["log_density"][6] < r["log_density"][0]
    True

    A new point in the middle of a cluster is normal; one far away is not:

    >>> r2 = geron_anomaly_gmm(X, n_components=2, contamination=0.1, seed=1,
    ...                        X_new=[[0.1], [500.0]])
    >>> r2["new_is_anomaly"]
    [False, True]

    Contamination outside (0, 1) is an error, not a clamp:

    >>> geron_anomaly_gmm(X, contamination=0.0)
    Traceback (most recent call last):
      ...
    ValueError: geron_anomaly_gmm: contamination must lie strictly in (0, 1), got 0.0

    References
    ----------
    Géron Ch 8
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_anomaly_gmm: X must be a non-empty (m, d) array, got shape {A.shape}")
    c = float(contamination)
    if not (0.0 < c < 1.0):
        raise ValueError(f"geron_anomaly_gmm: contamination must lie strictly in (0, 1), got {contamination!r}")

    fit = geron_gaussian_mixture(A, n_components=n_components, seed=seed)
    pi = np.asarray(fit["weights"], dtype=float)
    mu = np.asarray(fit["means"], dtype=float)
    Sig = np.asarray(fit["covariances"], dtype=float)

    def log_density(Z):
        lp = np.stack([np.log(pi[k] + 1e-300) + gmm_log_pdf(Z, mu[k], Sig[k]) for k in range(pi.size)], axis=1)
        mx = lp.max(axis=1, keepdims=True)
        return mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))

    ld = log_density(A)
    if threshold is None:
        log_thr = float(np.quantile(ld, c))
        thr = float(np.exp(log_thr))
    else:
        thr = float(threshold)
        if not np.isfinite(thr) or thr <= 0:
            raise ValueError(f"geron_anomaly_gmm: threshold must be a positive density, got {threshold!r}")
        log_thr = float(np.log(thr))
    flag = ld < log_thr

    new_ld = new_flag = None
    if X_new is not None:
        Z = np.atleast_2d(np.asarray(X_new, dtype=float))
        if Z.shape[1] != A.shape[1]:
            raise ValueError(f"geron_anomaly_gmm: X_new has {Z.shape[1]} features but X has {A.shape[1]}")
        new_ld = log_density(Z)
        new_flag = (new_ld < log_thr).tolist()
        new_ld = new_ld.tolist()

    return RichResult(
        title="GMM anomaly detection",
        summary_lines=[("Anomalies", int(flag.sum())), ("Threshold density", thr), ("Components", int(pi.size))],
        interpretation="Densities are compared in log space; the cut is set by the contamination quantile, not an absolute scale.",
        payload={
            "is_anomaly": flag.tolist(),
            "density": np.exp(ld).tolist(),
            "log_density": ld.tolist(),
            "threshold": thr,
            "log_threshold": log_thr,
            "n_anomalies": int(flag.sum()),
            "anomaly_indices": np.flatnonzero(flag).tolist(),
            "new_density": None if new_ld is None else np.exp(np.asarray(new_ld)).tolist(),
            "new_log_density": new_ld,
            "new_is_anomaly": new_flag,
            "contamination": c,
            "gmm": {"weights": pi.tolist(), "means": mu.tolist(), "log_likelihood": float(fit["log_likelihood"])},
            "estimate": float(flag.mean()),
            "n": int(A.shape[0]),
            "method": "GMM density thresholding in log space; mixture fitted by hmgmm",
        },
    )


def cheatsheet():
    return "hmgand: GMM-based anomaly detection: low-density points are anomalies"
