# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pretraining on auxiliary related task."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_auxiliary_task_pretraining"]


def _gd(theta, X, y, lr, epochs):
    losses = np.empty(epochs)
    n = X.shape[0]
    for e in range(epochs):
        resid = X @ theta - y
        losses[e] = float(resid @ resid / n)
        theta = theta - lr * (2.0 / n) * (X.T @ resid)
    return theta, losses


def geron_auxiliary_task_pretraining(model, aux_data, target_data, aux_epochs=200, epochs=20, lr=0.05):
    """
    Pretraining on an auxiliary related task.

    Formula: pretrain on task A (abundant labels), fine-tune on task B

    Runs the real two-stage schedule on a linear model: gradient descent on
    the auxiliary task, then the *same* parameter vector is fine-tuned on the
    (small) target task. The from-scratch control is trained on the target
    task alone with an identical budget, so the reported transfer gain is a
    measured difference rather than an assertion.

    Parameters
    ----------
    model : array-like or None
        Initial parameter vector shared by both arms; None means zeros.
    aux_data : tuple (X_aux, y_aux)
        Auxiliary task with abundant labels.
    target_data : tuple (X_target, y_target)
        Downstream task. Must have the same number of features as `aux_data`.
    aux_epochs : int
        Gradient steps on the auxiliary task (>= 1).
    epochs : int
        Fine-tuning steps on the target task (>= 1); the control gets the same.
    lr : float
        Step size (positive).

    Returns
    -------
    result : RichResult
        Keys: theta, theta_pretrained, target_loss, scratch_loss,
        transfer_gain, aux_losses, finetune_losses, estimate, n, method.

    Examples
    --------
    The auxiliary task shares the target's coefficients, so pretraining
    lands the fine-tuning arm below the from-scratch control:

    >>> Xa = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    >>> ya = [1.0, 2.0, 3.0, 4.0]
    >>> Xt = [[1.0, 1.0], [2.0, 0.0]]
    >>> yt = [3.0, 2.0]
    >>> r = geron_auxiliary_task_pretraining(None, (Xa, ya), (Xt, yt), aux_epochs=300, epochs=5)
    >>> bool(r["target_loss"] < r["scratch_loss"])
    True
    >>> bool(r["transfer_gain"] > 0)
    True
    >>> [round(float(t), 3) for t in r["theta_pretrained"]]
    [1.0, 2.0]

    References
    ----------
    Géron Ch 11
    """
    def _pair(name, data):
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            raise ValueError(f"geron_auxiliary_task_pretraining: {name} must be a (X, y) pair")
        Xd = np.asarray(data[0], dtype=float)
        if Xd.ndim == 1:
            Xd = Xd.reshape(-1, 1)
        yd = np.asarray(data[1], dtype=float).ravel()
        if Xd.ndim != 2 or Xd.shape[0] == 0:
            raise ValueError(f"geron_auxiliary_task_pretraining: {name} X must be a non-empty 2-D matrix")
        if yd.size != Xd.shape[0]:
            raise ValueError(
                f"geron_auxiliary_task_pretraining: {name} has {Xd.shape[0]} rows but {yd.size} targets"
            )
        if not (np.all(np.isfinite(Xd)) and np.all(np.isfinite(yd))):
            raise ValueError(f"geron_auxiliary_task_pretraining: {name} must be finite")
        return Xd, yd

    Xa, ya = _pair("aux_data", aux_data)
    Xt, yt = _pair("target_data", target_data)
    if Xa.shape[1] != Xt.shape[1]:
        raise ValueError(
            f"geron_auxiliary_task_pretraining: aux task has {Xa.shape[1]} features "
            f"but target task has {Xt.shape[1]}; transfer needs a shared representation"
        )
    d = Xa.shape[1]
    theta0 = np.zeros(d) if model is None else np.asarray(model, dtype=float).ravel()
    if theta0.size != d:
        raise ValueError(f"geron_auxiliary_task_pretraining: model has {theta0.size} parameters but {d} features")
    AE, FE = int(aux_epochs), int(epochs)
    if AE < 1 or FE < 1:
        raise ValueError("geron_auxiliary_task_pretraining: aux_epochs and epochs must both be >= 1")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("geron_auxiliary_task_pretraining: lr must be a positive finite step size")

    theta_pre, aux_losses = _gd(theta0.copy(), Xa, ya, step, AE)
    theta_ft, ft_losses = _gd(theta_pre.copy(), Xt, yt, step, FE)
    theta_scratch, scratch_losses = _gd(theta0.copy(), Xt, yt, step, FE)

    def mse(th):
        r = Xt @ th - yt
        return float(r @ r / yt.size)

    target_loss = mse(theta_ft)
    scratch_loss = mse(theta_scratch)

    return RichResult(
        title="Auxiliary-task pretraining",
        summary_lines=[
            ("Fine-tuned target MSE", target_loss),
            ("From-scratch target MSE", scratch_loss),
            ("Transfer gain", scratch_loss - target_loss),
        ],
        payload={
            "theta": theta_ft,
            "theta_pretrained": theta_pre,
            "theta_scratch": theta_scratch,
            "target_loss": target_loss,
            "scratch_loss": scratch_loss,
            "transfer_gain": scratch_loss - target_loss,
            "aux_losses": aux_losses,
            "finetune_losses": ft_losses,
            "scratch_losses": scratch_losses,
            "estimate": target_loss,
            "n": int(yt.size),
            "method": "Auxiliary-task pretraining then fine-tuning, against an equal-budget scratch control",
        },
    )


def cheatsheet():
    return "hmauxpt: Pretraining on auxiliary related task"
