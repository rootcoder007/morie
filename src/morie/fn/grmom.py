# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classical momentum optimizer step."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_momentum_update"]

_METHOD = "Momentum optimizer step"


def geron_momentum_update(theta, grad, v, eta, beta=0.9):
    r"""One momentum step.

    .. math::
        v_{t+1} = \beta v_t + g_t,\qquad
        \theta_{t+1} = \theta_t - \eta\, v_{t+1}

    On a constant gradient the velocity converges to
    :math:`g/(1-\beta)`, so momentum with :math:`\beta = 0.9` ends up
    moving *ten times* faster than plain gradient descent with the same
    ``eta``.  That terminal factor is reported as ``terminal_speedup``,
    and it is the reason a learning rate transplanted from SGD to
    momentum usually diverges.

    Unlike Adam (:mod:`morie.fn.gradmo`) there is no bias correction
    here: the source formula has none, and the first step is
    correspondingly small -- ``eta * g`` exactly, as in plain SGD.

    Parameters
    ----------
    theta, grad, v : array-like
        Parameters, gradient and velocity, all the same shape.
    eta : float
        Positive learning rate.
    beta : float, optional
        Momentum in ``[0, 1)``; ``beta = 0`` reduces to plain gradient
        descent. Default 0.9.

    Returns
    -------
    RichResult
        Payload keys ``theta_new``, ``v_new``, ``step``,
        ``terminal_speedup`` (:math:`1/(1-\beta)`), ``estimate`` (step
        L2 norm), ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Momentum Optimization section.

    Examples
    --------
    From rest the first step is exactly ``eta * g`` -- no momentum has
    built up yet:

    >>> r = geron_momentum_update([1.0], [0.1], [0.0], eta=0.1, beta=0.9)
    >>> r["v_new"]
    [0.1]
    >>> round(r["theta_new"][0], 10)
    0.99

    Feed the same gradient again and the velocity grows toward
    ``g/(1-beta) = 1``:

    >>> r2 = geron_momentum_update(r["theta_new"], [0.1], r["v_new"], eta=0.1)
    >>> round(r2["v_new"][0], 10)
    0.19
    >>> round(r2["terminal_speedup"], 6)
    10.0

    ``beta = 0`` is plain gradient descent:

    >>> geron_momentum_update([1.0], [0.5], [99.0], eta=1.0, beta=0.0)["theta_new"]
    [0.5]
    """
    theta = np.asarray(theta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    v = np.asarray(v, dtype=float)
    for name, arr in (("grad", grad), ("v", v)):
        if arr.shape != theta.shape:
            raise ValueError(f"{name} shape {arr.shape} != theta shape {theta.shape}.")
    if theta.size == 0:
        raise ValueError("theta is empty.")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(grad)) and np.all(np.isfinite(v))):
        raise ValueError("theta, grad and v must all be finite.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError(
            f"beta must lie in [0, 1); at beta = 1 the velocity never decays and "
            f"the optimizer cannot converge. Got {beta}."
        )

    v_new = beta * v + grad
    step = eta * v_new
    theta_new = theta - step

    return RichResult(
        title="Momentum update",
        summary_lines=[("Step L2", float(np.linalg.norm(step))),
                       ("beta", beta), ("Terminal speedup", 1.0 / (1.0 - beta))],
        payload={
            "theta_new": theta_new.tolist(),
            "v_new": v_new.tolist(),
            "step": step.tolist(),
            "terminal_speedup": 1.0 / (1.0 - beta),
            "beta": beta,
            "eta": eta,
            "estimate": float(np.linalg.norm(step)),
            "n": int(theta.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmom: v = beta*v + g; theta -= eta*v; terminal speed g/(1-beta)"
