# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct Preference Optimization loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_dpo_loss"]

_METHOD = "Direct Preference Optimization loss"


def geron_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=0.1):
    r"""Preference loss with no reward model in sight.

    .. math::
        L_{\text{DPO}} = -\log \sigma\Bigl(\beta\bigl[
        (\log \pi(y_w|x) - \log \pi_{\text{ref}}(y_w|x))
        - (\log \pi(y_l|x) - \log \pi_{\text{ref}}(y_l|x))\bigr]\Bigr)

    Everything hangs on the two *differences* from the reference model.
    A completion the reference already loved earns the policy no credit;
    what is optimised is how much more the policy prefers the chosen
    response than the reference did, relative to the rejected one.  That
    implicit reward is why DPO needs no separate reward network, and the
    :math:`\beta` is the KL leash: large ``beta`` punishes drift from
    the reference hard.

    ``-log sigma`` is computed as ``logaddexp(0, -x)``, which is exact
    for large negative margins where ``log(sigmoid(x))`` would underflow
    to ``-inf``.

    Parameters
    ----------
    logp_w, logp_l : array-like or float
        Policy log-probabilities of the preferred and rejected
        completions.
    logp_ref_w, logp_ref_l : array-like or float
        The same under the frozen reference model.
    beta : float, optional
        Positive KL strength, default 0.1.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``margin`` (the bracketed term),
        ``implicit_reward_chosen``, ``implicit_reward_rejected``,
        ``accuracy`` (fraction with a positive margin),
        ``per_pair_loss``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, DPO section (Rafailov et al. 2023).

    Examples
    --------
    A policy identical to the reference has zero margin and pays
    ``log 2`` -- it has expressed no preference at all:

    >>> r = geron_dpo_loss(0.0, 0.0, 0.0, 0.0, beta=1.0)
    >>> round(r["loss"], 10)
    0.6931471806
    >>> r["margin"]
    [0.0]

    Push the rejected completion down by 1 nat relative to the
    reference and the loss falls:

    >>> r2 = geron_dpo_loss(0.0, -1.0, 0.0, 0.0, beta=1.0)
    >>> r2["margin"]
    [1.0]
    >>> round(r2["loss"], 10)
    0.3132616875
    >>> r2["accuracy"]
    1.0

    Getting the preference backwards costs more than chance:

    >>> round(geron_dpo_loss(-1.0, 0.0, 0.0, 0.0, beta=1.0)["loss"], 10)
    1.3132616875
    """
    a = np.atleast_1d(np.asarray(logp_w, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(logp_l, dtype=float)).ravel()
    ra = np.atleast_1d(np.asarray(logp_ref_w, dtype=float)).ravel()
    rb = np.atleast_1d(np.asarray(logp_ref_l, dtype=float)).ravel()
    sizes = {a.size, b.size, ra.size, rb.size}
    if len(sizes) != 1:
        raise ValueError(
            f"all four log-probability arrays must have the same length, got "
            f"{a.size}, {b.size}, {ra.size}, {rb.size}."
        )
    for name, arr in (("logp_w", a), ("logp_l", b),
                      ("logp_ref_w", ra), ("logp_ref_l", rb)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite.")
        if np.any(arr > 0):
            raise ValueError(
                f"{name} holds log-probabilities and must be <= 0; got a positive value."
            )
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be a positive finite float, got {beta}.")

    rw = a - ra
    rl = b - rb
    margin = rw - rl
    per = np.logaddexp(0.0, -beta * margin)      # = -log sigmoid(beta * margin)
    loss = float(per.mean())

    return RichResult(
        title="DPO loss",
        summary_lines=[("Loss", loss), ("Mean margin", float(margin.mean())),
                       ("beta", beta)],
        payload={
            "loss": loss,
            "margin": margin.tolist(),
            "implicit_reward_chosen": (beta * rw).tolist(),
            "implicit_reward_rejected": (beta * rl).tolist(),
            "accuracy": float(np.mean(margin > 0)),
            "per_pair_loss": per.tolist(),
            "beta": beta,
            "estimate": loss,
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdpo: -log sigma(beta * [(lp_w - ref_w) - (lp_l - ref_l)]); implicit reward, no reward model"
