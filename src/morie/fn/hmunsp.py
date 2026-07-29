# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unsupervised pretraining: learn representation via reconstruction before labels."""

import numpy as np

from ._richresult import RichResult
from .hmaen import geron_autoencoder

__all__ = ["geron_unsupervised_pretraining"]


def _loo_mse(D, t):
    """Exact leave-one-out MSE of a least-squares fit via the hat matrix."""
    P = np.linalg.pinv(D.T @ D)
    theta = P @ (D.T @ t)
    resid = D @ theta - t
    h = np.clip(np.sum((D @ P) * D, axis=1), 0.0, 1.0)
    if np.any(h >= 1.0 - 1e-12):
        return theta, float("inf"), float(np.mean(resid * resid))
    return theta, float(np.mean((resid / (1.0 - h)) ** 2)), float(np.mean(resid * resid))


def geron_unsupervised_pretraining(X_unlab, X_lab, y_lab, bottleneck=1):
    """
    Unsupervised pretraining: learn representation via reconstruction before labels.

    Formula: pretrain autoencoder; use encoder weights as init

    The claim being tested is that a representation learned from the
    abundant *unlabeled* pool beats fitting the raw features on the scarce
    labeled set. So both arms are actually run:

    * pretrained arm -- an autoencoder is fitted on `X_unlab` alone
      (delegated to :func:`morie.fn.hmaen.geron_autoencoder`), the labeled
      inputs are pushed through that frozen encoder, and a linear head is
      fitted on the resulting codes;
    * control arm -- the same linear head fitted directly on the raw
      labeled features.

    Both are scored by exact leave-one-out MSE, which is the comparison
    that matters when the labeled set is small: the pretrained head has
    fewer parameters, so it pays less variance.

    Parameters
    ----------
    X_unlab : array-like
        Unlabeled pool (m, d).
    X_lab : array-like
        Labeled inputs (n, d), same width.
    y_lab : array-like
        Labels, length n.
    bottleneck : int, default 1
        Code width (1 <= bottleneck <= d).

    Returns
    -------
    result : RichResult
        Keys: encoder, codes, theta, pretrained_loo, control_loo, gain,
        recon_error, estimate, n, method.

    Examples
    --------
    The unlabeled pool lies on a line, so one code unit captures it; the
    labeled targets are a function of position along that line.

    >>> Xu = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
    >>> Xl = [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [5.0, 5.0]]
    >>> yl = [0.0, 2.0, 4.0, 5.0]
    >>> r = geron_unsupervised_pretraining(Xu, Xl, yl, bottleneck=1)
    >>> r["codes"].shape
    (4, 1)
    >>> round(float(r["recon_error"]), 12)
    0.0
    >>> bool(r["pretrained_loo"] < 1e-18)
    True

    With four noisy labeled points, the one-parameter pretrained head
    generalises better than the two-feature raw head:

    >>> Xl2 = [[0.0, 0.2], [2.0, 1.8], [4.0, 4.3], [5.0, 4.9]]
    >>> r2 = geron_unsupervised_pretraining(Xu, Xl2, [0.1, 2.0, 3.9, 5.2], bottleneck=1)
    >>> bool(r2["pretrained_loo"] < r2["control_loo"])
    True
    >>> bool(r2["gain"] > 0.0)
    True

    References
    ----------
    Géron Ch 11
    """
    U = np.asarray(X_unlab, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    L = np.asarray(X_lab, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    if U.ndim != 2 or U.size == 0:
        raise ValueError("geron_unsupervised_pretraining: X_unlab must be a non-empty (m, d) matrix")
    if L.ndim != 2 or L.size == 0:
        raise ValueError("geron_unsupervised_pretraining: X_lab must be a non-empty (n, d) matrix")
    if U.shape[1] != L.shape[1]:
        raise ValueError(
            f"geron_unsupervised_pretraining: X_unlab has {U.shape[1]} features but X_lab has {L.shape[1]}; "
            "pretraining transfers a representation, so the feature space must be shared"
        )
    t = np.asarray(y_lab, dtype=float).ravel()
    if t.size != L.shape[0]:
        raise ValueError(f"geron_unsupervised_pretraining: X_lab has {L.shape[0]} rows but y_lab has {t.size} labels")
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(L)) and np.all(np.isfinite(t))):
        raise ValueError("geron_unsupervised_pretraining: inputs must be finite")
    bn = int(bottleneck)
    if not (1 <= bn <= U.shape[1]):
        raise ValueError(f"geron_unsupervised_pretraining: bottleneck must lie in 1..{U.shape[1]}, got {bn}")

    ae = geron_autoencoder(U, bn)
    codes = np.asarray(ae["encode"](L), dtype=float)
    Dp = np.hstack([np.ones((codes.shape[0], 1)), codes])
    theta_p, loo_p, train_p = _loo_mse(Dp, t)
    Dc = np.hstack([np.ones((L.shape[0], 1)), L])
    theta_c, loo_c, train_c = _loo_mse(Dc, t)

    return RichResult(
        title="Unsupervised pretraining",
        summary_lines=[
            ("Unlabeled pool", int(U.shape[0])),
            ("Labeled examples", int(L.shape[0])),
            ("Code width", bn),
            ("Pretrained LOO MSE", loo_p),
            ("Raw-feature LOO MSE", loo_c),
        ],
        interpretation=(
            "Pretraining pays when labels are scarce and the unlabeled pool shares the same structure; "
            "if the encoder throws away the label-relevant direction, it costs instead."
        ),
        payload={
            "encoder": np.asarray(ae["encoder"], dtype=float),
            "encode": ae["encode"],
            "codes": codes,
            "theta": theta_p,
            "theta_control": theta_c,
            "pretrained_loo": loo_p,
            "control_loo": loo_c,
            "pretrained_train_mse": train_p,
            "control_train_mse": train_c,
            "gain": float(loo_c - loo_p),
            "recon_error": float(ae["recon_error"]),
            "explained_variance_ratio": np.asarray(ae["explained_variance_ratio"], dtype=float),
            "estimate": loo_p,
            "n": int(L.shape[0]),
            "method": "Autoencoder pretraining on the unlabeled pool (hmaen), frozen encoder + linear head, LOO-scored against a raw-feature control",
        },
    )


def cheatsheet():
    return "hmunsp: Unsupervised pretraining: learn representation via reconstruction before labels"
