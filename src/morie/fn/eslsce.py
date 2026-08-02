# morie.fn -- function file (rootcoder007/morie)
"""Score matching -- Hyvarinen (2005)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_score_match"]


def esl_score_match(score, X, grad_score=None, eps=1e-5):
    r"""Hyvarinen score-matching objective for an unnormalised model.

    Fitting an energy model needs :math:`\log Z(\theta)`, which is usually
    intractable. Score matching sidesteps it entirely by matching the
    *gradient of the log density with respect to x*, in which :math:`Z` --
    constant in :math:`x` -- has already cancelled:

    .. math::
        J(\theta) = \tfrac12 E_x\!\left\lVert
            \nabla_x \log q(x;\theta) - \nabla_x \log p(x)\right\rVert^2 .

    Hyvarinen's result is that this equals, up to a constant independent of
    :math:`\theta`, a quantity depending only on the model:

    .. math::
        J(\theta) = E_x\left[\operatorname{tr}\nabla_x \psi(x)
            + \tfrac12 \lVert \psi(x)\rVert^2\right] + \text{const},
        \qquad \psi = \nabla_x \log q .

    So the data's own score is never needed -- that is the entire trick, and
    it is why an unnormalised model can be fitted at all.

    The trace of the Jacobian costs ``d`` extra evaluations per point when
    ``grad_score`` is not supplied, which is what makes plain score matching
    expensive in high dimension and motivated the sliced and denoising
    variants.

    Parameters
    ----------
    score : callable
        :math:`\psi(X) = \nabla_x \log q(x;\theta)`, mapping ``(n, d)`` to
        ``(n, d)``.
    X : array-like
        Samples ``(n, d)``.
    grad_score : callable, optional
        Returns the diagonal of :math:`\nabla_x \psi` as ``(n, d)``. When
        omitted a central difference is used.
    eps : float
        Finite-difference step.

    Returns
    -------
    RichResult
        ``objective``, ``trace_term``, ``norm_term``, ``per_point``.

    References
    ----------
    Hyvarinen, A. (2005). Estimation of non-normalized statistical models by
        score matching. *JMLR*, 6, 695-709.

    Examples
    --------
    For a Gaussian model the objective is minimised at the true parameters.
    With ``q`` a ``N(mu, 1)`` density the score is ``-(x - mu)``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(2.0, 1.0, (4000, 1))
    >>> J = [esl_score_match(lambda Z, m=m: -(Z - m), X)["objective"]
    ...      for m in [0.0, 1.0, 2.0, 3.0]]
    >>> int(np.argmin(J))
    2

    The trace term is what penalises a score that is merely small: it is
    -1 per dimension for the unit Gaussian, independent of the data.

    >>> r = esl_score_match(lambda Z: -(Z - 2.0), X)
    >>> bool(abs(r["trace_term"] + 1.0) < 1e-3)
    True

    An analytic Jacobian diagonal agrees with the finite difference.

    >>> a = esl_score_match(lambda Z: -(Z - 2.0), X,
    ...                     grad_score=lambda Z: -np.ones_like(Z))["objective"]
    >>> bool(abs(a - r["objective"]) < 1e-3)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    psi = np.atleast_2d(np.asarray(score(X), dtype=float))
    if psi.shape != X.shape:
        raise ValueError(f"score returned shape {psi.shape}, expected {X.shape}")

    if grad_score is None:
        diag = np.empty_like(X)
        for j in range(d):
            Xp, Xm = X.copy(), X.copy()
            Xp[:, j] += eps
            Xm[:, j] -= eps
            diag[:, j] = (np.asarray(score(Xp))[:, j] - np.asarray(score(Xm))[:, j]) / (2 * eps)
    else:
        diag = np.atleast_2d(np.asarray(grad_score(X), dtype=float))
        if diag.shape != X.shape:
            raise ValueError(f"grad_score returned shape {diag.shape}, expected {X.shape}")

    per = diag.sum(axis=1) + 0.5 * (psi**2).sum(axis=1)
    return RichResult(
        title="Score matching objective",
        summary_lines=[("n", n), ("d", d), ("J", float(per.mean()))],
        payload={
            "objective": float(per.mean()),
            "trace_term": float(diag.sum(axis=1).mean()),
            "norm_term": float(0.5 * (psi**2).sum(axis=1).mean()),
            "per_point": per, "n": int(n), "d": int(d),
            "method": "esl_score_match",
        },
    )


def cheatsheet():
    return "eslsce: fits unnormalised models -- log Z cancels in grad_x, and the DATA score is never needed"
