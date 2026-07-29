# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Denoising score matching (NCSN) loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_score_matching_loss"]

_METHOD = "Denoising score matching (NCSN) loss"


def geron_score_matching_loss(x0, sigma, eps, score_pred, weight=None):
    r"""Squared error between the predicted score and the denoising target.

    .. math::
        L = \mathbb{E}_{\sigma, x_0, \epsilon}\left[
            \bigl\|\epsilon/\sigma - s_{\theta}(x_0 + \sigma\epsilon, \sigma)
            \bigr\|^2\right]

    Explicit score matching needs :math:`\nabla_x \log p(x)`, which
    nobody has.  The denoising form replaces it with something known: for
    a Gaussian perturbation the score of the *noised* density is exactly
    :math:`-\epsilon/\sigma`, and the noise is the quantity you drew
    yourself.  (The worklist states the target with a positive sign; the
    signed convention only flips the sign of :math:`s_\theta`, and both
    the target and the residual are returned here so the convention in
    use is never ambiguous.)

    Because the target scales as :math:`1/\sigma`, small noise levels
    dominate the loss unless it is reweighted -- NCSN's
    :math:`\lambda(\sigma) = \sigma^2` -- which is what ``weight``
    supplies.

    ``score_pred`` may be an array or a callable
    ``score_pred(x_noisy, sigma)``; the callable's output shape is
    enforced.

    Parameters
    ----------
    x0 : array-like, shape (m, n)
        Clean data.
    sigma : float or array-like, shape (m,)
        Noise scale(s), strictly positive.
    eps : array-like, same shape as ``x0``
    score_pred : array-like or callable
    weight : {None, "sigma2"} or array-like, optional

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``per_sample``, ``target``,
        ``x_noisy``, ``residual``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, Score-based Generative / NCSN section.

    Examples
    --------
    A perfect score network drives the loss to zero:

    >>> x0 = [[0.0]]
    >>> r = geron_score_matching_loss(x0, 0.5, [[1.0]], [[2.0]])
    >>> r["target"], r["loss"]
    ([[2.0]], 0.0)

    Predicting zero instead costs the target's squared norm, ``4``:

    >>> geron_score_matching_loss(x0, 0.5, [[1.0]], [[0.0]])["loss"]
    4.0

    Sigma-squared weighting removes the 1/sigma blow-up:

    >>> w = geron_score_matching_loss(x0, 0.5, [[1.0]], [[0.0]], weight="sigma2")
    >>> w["loss"]
    1.0
    """
    X = np.atleast_2d(np.asarray(x0, dtype=float))
    E = np.atleast_2d(np.asarray(eps, dtype=float))
    if X.size == 0:
        raise ValueError("x0 is empty.")
    if E.shape != X.shape:
        raise ValueError(f"eps has shape {E.shape} but x0 has {X.shape}.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(E)):
        raise ValueError("x0 and eps must be finite.")
    s = np.asarray(sigma, dtype=float).ravel()
    if s.size == 1:
        s = np.full(X.shape[0], float(s[0]))
    if s.size != X.shape[0]:
        raise ValueError(f"sigma has {s.size} entries but x0 has {X.shape[0]} rows.")
    if np.any(s <= 0) or not np.all(np.isfinite(s)):
        raise ValueError(f"sigma must be strictly positive and finite, got minimum {float(s.min())}.")

    x_noisy = X + s[:, None] * E
    target = E / s[:, None]

    if callable(score_pred):
        S = np.atleast_2d(np.asarray(score_pred(x_noisy, s), dtype=float))
    else:
        S = np.atleast_2d(np.asarray(score_pred, dtype=float))
    if S.shape != X.shape:
        raise ValueError(f"score_pred has shape {S.shape} but x0 has {X.shape}.")
    if not np.all(np.isfinite(S)):
        raise ValueError("score_pred contains non-finite values.")

    resid = target - S
    per = np.sum(resid**2, axis=1)
    if weight is None:
        w = np.ones(X.shape[0])
    elif isinstance(weight, str):
        if weight != "sigma2":
            raise ValueError(f"weight must be None, 'sigma2' or an array, got {weight!r}.")
        w = s**2
    else:
        w = np.asarray(weight, dtype=float).ravel()
        if w.size != X.shape[0]:
            raise ValueError(f"weight has {w.size} entries but x0 has {X.shape[0]} rows.")
        if np.any(w < 0):
            raise ValueError("weight must be non-negative.")
    per_w = w * per
    loss = float(per_w.mean())

    return RichResult(
        title="Score matching loss",
        summary_lines=[("Loss", loss), ("Samples", int(X.shape[0]))],
        payload={
            "loss": loss,
            "per_sample": per_w.tolist(),
            "target": target.tolist(),
            "x_noisy": x_noisy.tolist(),
            "residual": resid.tolist(),
            "estimate": loss,
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grscm: L = ||eps/sigma - s_theta(x0+sigma eps)||^2; weight='sigma2' balances the noise levels"
