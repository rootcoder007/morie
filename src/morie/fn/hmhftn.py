# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face Trainer API high-level training loop."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_hf_trainer"]

_METHOD = "Trainer loop (mini-batch SGD with per-epoch evaluation)"


def geron_hf_trainer(model, args, train_ds, eval_ds=None):
    """
    Hugging Face Trainer API high-level training loop.

    Formula: Trainer(model, args, train_ds, eval_ds).train()

    The Trainer is a loop, not a model, and the loop is what is
    implemented here natively: shuffle, iterate mini-batches, compute
    loss and gradient, step, evaluate at the end of each epoch, keep the
    best checkpoint by evaluation loss.

    ``model`` is a mapping supplying its own parameters and objective:

    ``model["params"]``
        1-D array of parameters (copied, not mutated).
    ``model["loss_and_grad"]``
        ``f(params, X, y) -> (loss, grad)`` with ``grad`` the same shape
        as ``params``.

    ``args`` is a mapping of ``epochs``, ``batch_size``,
    ``learning_rate``, ``seed``.  The contract is enforced on every
    call: a gradient whose shape drifts from the parameters, or a
    non-finite loss, raises with the epoch and step named instead of
    quietly poisoning the weights.

    ``best_params`` is the checkpoint with the lowest *evaluation* loss,
    which is not in general the final one -- returning the last epoch's
    weights is how a training loop hands back an overfitted model.

    Parameters
    ----------
    model : mapping
        With keys ``params`` and ``loss_and_grad`` as above.
    args : mapping
        ``epochs`` (default 1), ``batch_size`` (default 8),
        ``learning_rate`` (default 0.01), ``seed`` (default 0).
    train_ds : (X, y) pair
        Training features and targets.
    eval_ds : (X, y) pair, optional
        Evaluation set; without it the training loss is used for the
        checkpoint criterion.

    Returns
    -------
    result : RichResult
        Keys: params, best_params, train_loss, eval_loss, history,
        best_epoch, estimate, n, method.

    Examples
    --------
    Least squares on ``y = 3x`` from a standing start, with the exact
    MSE gradient supplied by the caller:

    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([3.0, 6.0, 9.0, 12.0])
    >>> def lg(p, Xb, yb):
    ...     r = Xb @ p - yb
    ...     return float(np.mean(r ** 2)), (2.0 / len(yb)) * (Xb.T @ r)
    >>> m = {"params": np.zeros(1), "loss_and_grad": lg}
    >>> out = geron_hf_trainer(m, {"epochs": 200, "batch_size": 2, "learning_rate": 0.02},
    ...                        (X, y), (X, y))
    >>> bool(abs(float(out["params"][0]) - 3.0) < 1e-3)
    True

    The loss falls and the history has one entry per epoch:

    >>> len(out["history"])
    200
    >>> bool(out["history"][-1]["train_loss"] < out["history"][0]["train_loss"])
    True

    The best checkpoint is chosen on evaluation loss:

    >>> best = min(h["eval_loss"] for h in out["history"])
    >>> bool(abs(out["eval_loss"] - best) < 1e-12)
    True

    A gradient of the wrong shape is caught immediately:

    >>> bad = {"params": np.zeros(1), "loss_and_grad": lambda p, Xb, yb: (1.0, np.zeros(3))}
    >>> geron_hf_trainer(bad, {"epochs": 1, "batch_size": 4}, (X, y))
    Traceback (most recent call last):
        ...
    ValueError: geron_hf_trainer: loss_and_grad returned a gradient of shape (3,) at epoch 1 step 1, but params has shape (1,)

    References
    ----------
    Géron Ch 14
    """
    if not hasattr(model, "__getitem__"):
        raise ValueError("geron_hf_trainer: model must be a mapping with keys 'params' and 'loss_and_grad'")
    try:
        params = np.atleast_1d(np.asarray(model["params"], dtype=float)).ravel().copy()
    except (KeyError, TypeError):
        raise ValueError("geron_hf_trainer: model is missing 'params'") from None
    try:
        lg = model["loss_and_grad"]
    except (KeyError, TypeError):
        raise ValueError("geron_hf_trainer: model is missing 'loss_and_grad'") from None
    if not callable(lg):
        raise ValueError(f"geron_hf_trainer: model['loss_and_grad'] must be callable, got {type(lg).__name__}")
    if params.size == 0:
        raise ValueError("geron_hf_trainer: model['params'] is empty")

    cfg = dict(args) if args is not None else {}
    epochs = int(cfg.get("epochs", 1))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("learning_rate", 0.01))
    seed = int(cfg.get("seed", 0))
    if epochs < 1:
        raise ValueError(f"geron_hf_trainer: args['epochs'] must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"geron_hf_trainer: args['batch_size'] must be at least 1, got {batch_size}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_hf_trainer: args['learning_rate'] must be positive and finite, got {lr}")

    def _dataset(ds, name):
        try:
            Xd, yd = ds
        except (TypeError, ValueError):
            raise ValueError(f"geron_hf_trainer: {name} must be an (X, y) pair") from None
        Xd = np.atleast_2d(np.asarray(Xd, dtype=float))
        yd = np.asarray(yd, dtype=float).ravel()
        if Xd.shape[0] != yd.size:
            raise ValueError(f"geron_hf_trainer: {name} has {Xd.shape[0]} rows of X but {yd.size} targets")
        if Xd.size == 0:
            raise ValueError(f"geron_hf_trainer: {name} is empty")
        return Xd, yd

    Xtr, ytr = _dataset(train_ds, "train_ds")
    if eval_ds is None:
        Xev, yev = Xtr, ytr
        have_eval = False
    else:
        Xev, yev = _dataset(eval_ds, "eval_ds")
        have_eval = True

    m = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    history = []
    best_params = params.copy()
    best_loss = np.inf
    best_epoch = 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(m)
        losses = []
        step = 0
        for start in range(0, m, batch_size):
            step += 1
            idx = order[start : start + batch_size]
            loss, grad = lg(params, Xtr[idx], ytr[idx])
            grad = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
            if grad.shape != params.shape:
                raise ValueError(
                    f"geron_hf_trainer: loss_and_grad returned a gradient of shape {grad.shape} "
                    f"at epoch {ep} step {step}, but params has shape {params.shape}"
                )
            loss = float(loss)
            if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
                raise ValueError(
                    f"geron_hf_trainer: loss_and_grad returned a non-finite loss or gradient at epoch {ep} step {step}"
                )
            losses.append(loss)
            params = params - lr * grad

        train_loss = float(np.mean(losses))
        eval_loss, _ = lg(params, Xev, yev)
        eval_loss = float(eval_loss)
        if not np.isfinite(eval_loss):
            raise ValueError(f"geron_hf_trainer: evaluation loss is not finite at epoch {ep}")
        history.append({"epoch": ep, "train_loss": train_loss, "eval_loss": eval_loss})
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_params = params.copy()
            best_epoch = ep

    return RichResult(
        title="Trainer",
        summary_lines=[
            ("Epochs", epochs),
            ("Final train loss", history[-1]["train_loss"]),
            ("Best eval loss", best_loss),
            ("Best epoch", best_epoch),
        ],
        warnings=(
            [] if have_eval else ["no eval_ds was supplied, so the checkpoint criterion is the training loss."]
        ),
        interpretation=(
            "best_params is the lowest-evaluation-loss checkpoint, not the final weights; "
            "returning the last epoch is how a loop hands back an overfitted model."
        ),
        payload={
            "params": params,
            "best_params": best_params,
            "train_loss": history[-1]["train_loss"],
            "eval_loss": best_loss,
            "history": history,
            "best_epoch": best_epoch,
            "estimate": float(best_loss),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhftn: Trainer loop -- mini-batch SGD, per-epoch eval, best-checkpoint selection, enforced grad contract"


# compact alias per ledger/NAMING.md
geronhftrainer = geron_hf_trainer
