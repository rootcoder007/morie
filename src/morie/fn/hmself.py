# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-supervised learning: generate labels from the data itself via pretext task."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_self_supervised"]


def _fit_predict(A, t):
    D = np.hstack([np.ones((A.shape[0], 1)), A])
    theta = np.linalg.pinv(D.T @ D) @ (D.T @ t)
    return theta, D @ theta


def geron_self_supervised(X, pretext="mask", noise=0.1, seed=0):
    """
    Self-supervised learning: generate labels from the data itself via pretext task.

    Formula: L = E[(f(x_pretext) - y_pretext)^2]

    No labels are consumed anywhere in this function -- that is the point.
    The supervision is manufactured from `X`:

    * ``"mask"`` -- hold out one feature and predict it from the rest,
      once per feature (the tabular analogue of masked language
      modelling). A feature that is a deterministic function of the
      others is recovered exactly, and its pretext loss is zero.
    * ``"denoise"`` -- corrupt the inputs with deterministic LCG noise and
      reconstruct the clean values.

    A callable may be supplied instead, with the contract
    ``pretext(X) -> (X_pretext, y_pretext)`` where the two returns have
    matching row counts; anything else is an error rather than a
    broadcast.

    Parameters
    ----------
    X : array-like
        Unlabeled data (n, d).
    pretext : {"mask", "denoise"} or callable, default "mask"
        Pretext task.
    noise : float, default 0.1
        Corruption scale for ``"denoise"`` (>= 0).
    seed : int, default 0
        LCG seed for the corruption.

    Returns
    -------
    result : RichResult
        Keys: loss, task_losses, predictions, targets, r2, estimate,
        n, method.

    Examples
    --------
    The third column is the sum of the first two, so masking it is solved
    exactly while masking either of the others is not:

    >>> X = [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 5.0, 8.0], [0.0, 4.0, 4.0]]
    >>> r = geron_self_supervised(X, "mask")
    >>> round(float(r["task_losses"][2]), 12)
    0.0
    >>> bool(r["loss"] < 1e-20)
    True

    Denoising: the reconstruction error is bounded by the corruption that
    was injected.

    >>> r2 = geron_self_supervised(X, "denoise", noise=0.5)
    >>> bool(r2["loss"] > 0.0)
    True
    >>> r2["predictions"].shape
    (4, 3)

    References
    ----------
    Géron Ch 1
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_self_supervised: X must be a non-empty (n, d) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_self_supervised: X contains non-finite values")
    n, d = A.shape

    if callable(pretext):
        out = pretext(A)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise ValueError("geron_self_supervised: a callable pretext must return (X_pretext, y_pretext)")
        Xp = np.asarray(out[0], dtype=float)
        if Xp.ndim == 1:
            Xp = Xp.reshape(-1, 1)
        yp = np.asarray(out[1], dtype=float)
        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if Xp.shape[0] != yp.shape[0]:
            raise ValueError(
                f"geron_self_supervised: pretext returned {Xp.shape[0]} inputs but {yp.shape[0]} targets"
            )
        preds = np.empty_like(yp)
        losses = np.empty(yp.shape[1])
        for j in range(yp.shape[1]):
            _, p = _fit_predict(Xp, yp[:, j])
            preds[:, j] = p
            losses[j] = float(np.mean((p - yp[:, j]) ** 2))
        targets = yp
        name = "callable"
    else:
        task = str(pretext).lower()
        if task == "mask":
            if d < 2:
                raise ValueError("geron_self_supervised: the mask pretext needs at least 2 features")
            preds = np.empty_like(A)
            losses = np.empty(d)
            for j in range(d):
                keep = [c for c in range(d) if c != j]
                _, p = _fit_predict(A[:, keep], A[:, j])
                preds[:, j] = p
                losses[j] = float(np.mean((p - A[:, j]) ** 2))
            targets = A
            name = "mask"
        elif task == "denoise":
            sc = float(noise)
            if not np.isfinite(sc) or sc < 0:
                raise ValueError(f"geron_self_supervised: noise must be non-negative and finite, got {sc}")
            s = int(seed) % 2**32
            E = np.empty(n * d)
            for i in range(n * d):
                s = (1664525 * s + 1013904223) % 2**32
                E[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * sc
            Xn = A + E.reshape(n, d)
            preds = np.empty_like(A)
            losses = np.empty(d)
            for j in range(d):
                _, p = _fit_predict(Xn, A[:, j])
                preds[:, j] = p
                losses[j] = float(np.mean((p - A[:, j]) ** 2))
            targets = A
            name = "denoise"
        else:
            raise ValueError(f"geron_self_supervised: pretext must be 'mask', 'denoise' or a callable, got {pretext!r}")

    loss = float(np.mean(losses))
    var = np.var(targets, axis=0)
    r2 = np.where(var > 0, 1.0 - losses / np.where(var > 0, var, 1.0), np.nan)

    return RichResult(
        title="Self-supervised pretext task",
        summary_lines=[("Rows", n), ("Pretext", name), ("Mean pretext loss", loss)],
        interpretation=(
            "The labels are free because they were already in the data; a pretext task is useful only "
            "when solving it forces the model to learn structure the downstream task also needs."
        ),
        payload={
            "loss": loss,
            "task_losses": losses,
            "predictions": preds,
            "targets": targets,
            "r2": r2,
            "pretext": name,
            "estimate": loss,
            "n": int(n),
            "method": f"Self-supervised '{name}' pretext solved by least squares; no external labels used",
        },
    )


def cheatsheet():
    return "hmself: Self-supervised learning: generate labels from the data itself via pretext task"
