# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO per-grid-cell localization, objectness and classification loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_yolo_grid_loss"]

_METHOD = "YOLO grid loss (coord + obj + noobj + class)"


def geron_yolo_grid_loss(predictions, targets, lam_coord=5.0, lam_noobj=0.5):
    r"""Sum-squared detection loss over a grid.

    .. math::
        L = \sum_i \mathbb{1}^{\text{obj}}_i\bigl[
                L_{\text{bbox}}(i) + L_{\text{obj}}(i) + L_{\text{class}}(i)\bigr]
            + \lambda_{\text{noobj}}\sum_i \mathbb{1}^{\text{noobj}}_i
              L_{\text{noobj}}(i)

    with :math:`L_{\text{bbox}} = \lambda_{\text{coord}}[(x-\hat x)^2 +
    (y-\hat y)^2 + (\sqrt w - \sqrt{\hat w})^2 +
    (\sqrt h - \sqrt{\hat h})^2]`.

    Three deliberate asymmetries, all of them load-bearing.  The two
    lambdas exist because most cells are empty: without
    :math:`\lambda_{\text{noobj}} = 0.5` the background swamps the
    gradient, and without :math:`\lambda_{\text{coord}} = 5` localisation
    is drowned by classification.  The square roots on width and height
    make a 10-pixel error matter more on a small box than a large one,
    which plain squared error would not.  And everything except
    objectness is masked to cells that contain an object -- an empty cell
    has no box to regress and no class to name.

    Parameters
    ----------
    predictions, targets : array-like, shape (S, S, 5 + C)
        Channels ``[x, y, w, h, confidence, class probabilities...]``.
        In ``targets``, confidence is the object indicator (1 or 0).
    lam_coord, lam_noobj : float, optional
        Non-negative weights (YOLOv1 uses 5 and 0.5).

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``loss_coord``, ``loss_obj``,
        ``loss_noobj``, ``loss_class``, ``n_objects``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 12, YOLO section.

    Examples
    --------
    One 1x1 cell, one class.  A perfect prediction costs nothing:

    >>> p = [[[0.5, 0.5, 0.25, 0.25, 1.0, 1.0]]]
    >>> geron_yolo_grid_loss(p, p)["loss"]
    0.0

    Shift x by 0.1: the coordinate term is ``5 * 0.01 = 0.05``.

    >>> q = [[[0.6, 0.5, 0.25, 0.25, 1.0, 1.0]]]
    >>> r = geron_yolo_grid_loss(q, p)
    >>> round(r["loss_coord"], 10), round(r["loss"], 10)
    (0.05, 0.05)

    An empty cell only pays the objectness term, down-weighted:
    ``0.5 * 0.2^2 = 0.02``.

    >>> t0 = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    >>> p0 = [[[0.9, 0.9, 0.5, 0.5, 0.2, 0.7]]]
    >>> e = geron_yolo_grid_loss(p0, t0)
    >>> round(e["loss_noobj"], 10), round(e["loss"], 10)
    (0.02, 0.02)
    """
    P = np.asarray(predictions, dtype=float)
    T = np.asarray(targets, dtype=float)
    if P.ndim != 3 or P.shape[2] < 5:
        raise ValueError(
            f"predictions must be (S, S, 5 + C) with C >= 0, got shape {P.shape}."
        )
    if P.shape != T.shape:
        raise ValueError(f"targets has shape {T.shape} but predictions has {P.shape}.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(T)):
        raise ValueError("predictions and targets must be finite.")
    lam_coord = float(lam_coord)
    lam_noobj = float(lam_noobj)
    if lam_coord < 0 or lam_noobj < 0:
        raise ValueError(
            f"lam_coord and lam_noobj must be non-negative, got {lam_coord} and {lam_noobj}."
        )
    if np.any(P[..., 2:4] < 0) or np.any(T[..., 2:4] < 0):
        raise ValueError("box width and height must be non-negative; sqrt is taken of them.")
    conf_t = T[..., 4]
    if np.any((conf_t != 0) & (conf_t != 1)):
        raise ValueError("target confidence is the object indicator and must be 0 or 1.")

    obj = conf_t == 1
    noobj = ~obj

    dxy = ((P[..., 0:2] - T[..., 0:2]) ** 2).sum(axis=2)
    dwh = ((np.sqrt(P[..., 2:4]) - np.sqrt(T[..., 2:4])) ** 2).sum(axis=2)
    loss_coord = float(lam_coord * (dxy[obj].sum() + dwh[obj].sum()))
    dconf = (P[..., 4] - T[..., 4]) ** 2
    loss_obj = float(dconf[obj].sum())
    loss_noobj = float(lam_noobj * dconf[noobj].sum())
    if P.shape[2] > 5:
        dcls = ((P[..., 5:] - T[..., 5:]) ** 2).sum(axis=2)
        loss_class = float(dcls[obj].sum())
    else:
        loss_class = 0.0
    loss = loss_coord + loss_obj + loss_noobj + loss_class

    return RichResult(
        title="YOLO grid loss",
        summary_lines=[("Loss", loss), ("Cells with objects", int(obj.sum()))],
        payload={
            "loss": loss,
            "loss_coord": loss_coord,
            "loss_obj": loss_obj,
            "loss_noobj": loss_noobj,
            "loss_class": loss_class,
            "n_objects": int(obj.sum()),
            "n_cells": int(P.shape[0] * P.shape[1]),
            "estimate": loss,
            "n": int(P.shape[0] * P.shape[1]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gryol: lam_coord*(xy + sqrt-wh) + obj + lam_noobj*noobj + class, all but objectness masked to obj cells"
