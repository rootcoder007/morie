# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fine-tune a pretrained language model on a downstream task."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_finetune_lm"]


def geron_finetune_lm(
    model,
    dataset,
    epochs=10,
    lr=0.01,
    theta=None,
    freeze=None,
    batch_size=None,
    weight_decay=0.0,
    warmup=0,
):
    """
    Fine-tune a pretrained language model on a downstream task.

    Formula: theta <- theta - eta * grad L_task(theta; D_task)

    The optimiser is implemented natively and the model is a caller-
    supplied callable, so any differentiable model can be fine-tuned:
    ``model(theta, batch) -> (loss, grad)`` with ``grad`` the same shape
    as ``theta``. The contract is enforced on every call -- a model
    returning a scalar gradient, the wrong shape, or a non-finite loss
    raises rather than silently corrupting the parameters.

    What makes this *fine-tuning* rather than plain SGD is here too:

    * ``theta`` starts from the pretrained weights, not from noise, and
      ``drift`` reports how far training moved from that starting point --
      the quantity that catastrophic forgetting shows up in;
    * ``freeze`` is a boolean mask of parameters held fixed (the frozen
      lower layers), whose gradients are zeroed before the step;
    * ``warmup`` linearly ramps the learning rate over the first steps,
      standard practice because a large first step on pretrained weights
      destroys them;
    * ``weight_decay`` pulls towards zero as usual.

    Parameters
    ----------
    model : callable
        ``model(theta, batch) -> (loss, grad)``.
    dataset : sequence
        Training examples; batched in order.
    epochs : int, default 10
    lr : float, default 0.01
    theta : array-like, optional
        Pretrained parameters; default a single zero.
    freeze : array-like of bool, optional
        True marks a frozen parameter.
    batch_size : int, optional
        Default: the whole dataset per step.
    weight_decay : float, default 0.0
    warmup : int, default 0
        Steps of linear learning-rate warmup.

    Returns
    -------
    result : RichResult
        Keys: theta, theta_init, loss_history, drift, n_steps,
        lr_schedule, frozen, grad_norms, estimate, n, method.

    Examples
    --------
    A quadratic task ``L(theta) = (theta - 3)^2`` starting from the
    pretrained value 0: one full step of size 0.1 moves theta by
    ``0.1 * 2 * (0 - 3) = -0.6``, i.e. to 0.6.

    >>> def task(th, batch):
    ...     import numpy as np
    ...     return float((th[0] - 3.0) ** 2), np.array([2.0 * (th[0] - 3.0)])
    >>> r = geron_finetune_lm(task, [1], epochs=1, lr=0.1, theta=[0.0])
    >>> round(r["theta"][0], 12)
    0.6
    >>> round(r["loss_history"][0], 12)
    9.0

    Training converges to the optimum and the drift records the distance
    travelled from the pretrained weights:

    >>> r2 = geron_finetune_lm(task, [1], epochs=100, lr=0.1, theta=[0.0])
    >>> round(r2["theta"][0], 6)
    3.0
    >>> round(r2["drift"], 6)
    3.0

    A frozen parameter never moves, whatever its gradient:

    >>> def two(th, batch):
    ...     import numpy as np
    ...     return float(th[0] ** 2 + th[1] ** 2), np.array([2 * th[0], 2 * th[1]])
    >>> r3 = geron_finetune_lm(two, [1], epochs=5, lr=0.1, theta=[1.0, 1.0], freeze=[True, False])
    >>> r3["theta"][0]
    1.0
    >>> r3["theta"][1] < 1.0
    True

    Warmup ramps the rate linearly over the first steps:

    >>> r4 = geron_finetune_lm(task, [1], epochs=4, lr=0.1, theta=[0.0], warmup=2)
    >>> [round(v, 6) for v in r4["lr_schedule"]]
    [0.05, 0.1, 0.1, 0.1]

    References
    ----------
    Géron Ch 14
    """
    if not callable(model):
        raise ValueError("geron_finetune_lm: model must be a callable model(theta, batch) -> (loss, grad)")
    if dataset is None or len(dataset) == 0:
        raise ValueError("geron_finetune_lm: dataset is empty")
    th = np.atleast_1d(np.asarray([0.0] if theta is None else theta, dtype=float)).copy()
    if not np.all(np.isfinite(th)):
        raise ValueError("geron_finetune_lm: theta contains non-finite values")
    init = th.copy()
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_finetune_lm: epochs must be >= 1, got {epochs!r}")
    base_lr = float(lr)
    if not np.isfinite(base_lr) or base_lr < 0:
        raise ValueError(f"geron_finetune_lm: lr must be non-negative and finite, got {lr!r}")
    wd = float(weight_decay)
    if wd < 0:
        raise ValueError(f"geron_finetune_lm: weight_decay must be non-negative, got {weight_decay!r}")
    W = int(warmup)
    if W < 0:
        raise ValueError(f"geron_finetune_lm: warmup must be non-negative, got {warmup!r}")
    N = len(dataset)
    bs = N if batch_size is None else int(batch_size)
    if not (1 <= bs <= N):
        raise ValueError(f"geron_finetune_lm: batch_size must lie in 1..{N}, got {batch_size!r}")
    if freeze is None:
        mask = np.zeros(th.size, dtype=bool)
    else:
        mask = np.asarray(freeze, dtype=bool).ravel()
        if mask.size != th.size:
            raise ValueError(f"geron_finetune_lm: freeze has {mask.size} entries but theta has {th.size}")
        if mask.all():
            raise ValueError("geron_finetune_lm: every parameter is frozen; there is nothing to fine-tune")

    hist, sched, gnorms = [], [], []
    step = 0
    pos = 0
    for _ in range(E):
        batch = [dataset[(pos + i) % N] for i in range(bs)]
        pos = (pos + bs) % N
        out = model(th, batch)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise ValueError("geron_finetune_lm: model must return exactly (loss, grad)")
        loss, grad = out
        loss = float(loss)
        grad = np.atleast_1d(np.asarray(grad, dtype=float))
        if grad.shape != th.shape:
            raise ValueError(f"geron_finetune_lm: model returned a gradient of shape {grad.shape}, expected {th.shape}")
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise ValueError(f"geron_finetune_lm: model returned non-finite values at step {step}")
        step += 1
        cur_lr = base_lr * (step / W) if W and step <= W else base_lr
        g = grad + wd * th
        g = np.where(mask, 0.0, g)
        th = th - cur_lr * g
        hist.append(loss)
        sched.append(float(cur_lr))
        gnorms.append(float(np.linalg.norm(g)))

    return RichResult(
        title="Language model fine-tuning",
        summary_lines=[("Steps", step), ("Final loss", hist[-1]), ("Drift", float(np.linalg.norm(th - init)))],
        interpretation="Fine-tuning starts from pretrained weights; `drift` is the distance travelled away from them.",
        payload={
            "theta": th.tolist(),
            "theta_init": init.tolist(),
            "loss_history": hist,
            "grad_norms": gnorms,
            "lr_schedule": sched,
            "drift": float(np.linalg.norm(th - init)),
            "n_steps": int(step),
            "frozen": mask.tolist(),
            "n_frozen": int(mask.sum()),
            "weight_decay": wd,
            "warmup": W,
            "estimate": float(hist[-1]),
            "n": int(N),
            "method": "native SGD fine-tuning loop with frozen parameters, warmup and weight decay",
        },
    )


def cheatsheet():
    return "hmfth: Fine-tune a pretrained language model on a downstream task"
