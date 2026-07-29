# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Autoencoder anomaly detection: high reconstruction error indicates anomaly."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_anomaly_autoencoder"]


def geron_anomaly_autoencoder(model, X, threshold=None, quantile=0.99):
    """
    Autoencoder anomaly detection: high reconstruction error indicates anomaly.

    Formula: anomaly if ||x - decode(encode(x))||^2 > threshold

    Parameters
    ----------
    model : callable or object
        Either ``model(X) -> X_hat`` or an object exposing ``predict`` /
        ``reconstruct``, or a pair of ``encode`` / ``decode`` methods. The
        reconstruction must have exactly the shape of `X`; anything else is
        an error rather than something to broadcast around.
    X : array-like, shape (n, d)
        Data to score.
    threshold : float, optional
        Squared-error cut-off. When omitted it is set to the empirical
        `quantile` of the errors, which calibrates the detector on the data
        being scored (use an explicit threshold fitted on clean data when you
        have one).
    quantile : float, default 0.99
        Quantile in (0, 1] used when `threshold` is None.

    Returns
    -------
    result : RichResult
        Keys: errors, is_anomaly, threshold, n_anomalies, reconstruction,
        estimate, n, method.

    Examples
    --------
    >>> zero = lambda A: np.zeros_like(np.asarray(A, dtype=float))
    >>> r = geron_anomaly_autoencoder(zero, [[0.0], [3.0]], threshold=1.0)
    >>> [float(e) for e in r["errors"]]
    [0.0, 9.0]
    >>> [bool(b) for b in r["is_anomaly"]]
    [False, True]
    >>> r["n_anomalies"]
    1
    >>> r2 = geron_anomaly_autoencoder(zero, [[3.0], [4.0]], threshold=25.0)
    >>> r2["n_anomalies"]
    0

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_anomaly_autoencoder: X must be 2-D, got ndim={A.ndim}")
    if A.shape[0] == 0:
        raise ValueError("geron_anomaly_autoencoder: X has no rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_anomaly_autoencoder: X must be finite")

    if callable(model):
        recon = model(A)
    elif hasattr(model, "reconstruct"):
        recon = model.reconstruct(A)
    elif hasattr(model, "predict"):
        recon = model.predict(A)
    elif hasattr(model, "encode") and hasattr(model, "decode"):
        recon = model.decode(model.encode(A))
    else:
        raise ValueError(
            "geron_anomaly_autoencoder: model must be callable or expose "
            "reconstruct/predict, or encode+decode"
        )
    recon = np.asarray(recon, dtype=float)
    if recon.ndim == 1:
        recon = recon.reshape(-1, 1)
    if recon.shape != A.shape:
        raise ValueError(
            f"geron_anomaly_autoencoder: model returned shape {recon.shape} but X has shape {A.shape}"
        )
    if not np.all(np.isfinite(recon)):
        raise ValueError("geron_anomaly_autoencoder: model returned non-finite reconstructions")

    errors = np.sum((A - recon) ** 2, axis=1)

    if threshold is None:
        q = float(quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"geron_anomaly_autoencoder: quantile must lie in (0, 1], got {q}")
        thr = float(np.quantile(errors, q))
        calibrated = True
    else:
        thr = float(threshold)
        if not np.isfinite(thr) or thr < 0:
            raise ValueError("geron_anomaly_autoencoder: threshold must be a finite non-negative squared error")
        calibrated = False

    flags = errors > thr
    n_anom = int(np.sum(flags))

    return RichResult(
        title="Autoencoder anomaly detection",
        summary_lines=[("Threshold", thr), ("Anomalies", n_anom), ("Median error", float(np.median(errors)))],
        payload={
            "errors": errors,
            "is_anomaly": flags,
            "threshold": thr,
            "threshold_calibrated": calibrated,
            "n_anomalies": n_anom,
            "reconstruction": recon,
            "estimate": float(np.mean(errors)),
            "n": int(A.shape[0]),
            "method": "Autoencoder anomaly detection by squared reconstruction error",
        },
    )


def cheatsheet():
    return "hmanae: Autoencoder anomaly detection: high reconstruction error indicates anomaly"
