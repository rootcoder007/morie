# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prioritized experience replay importance-sampling weights."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_prioritized_experience_weight"]

_METHOD = "Prioritized experience replay IS weights"


def geron_prioritized_experience_weight(priorities, N=None, alpha=0.6, beta=0.4, eps=1e-6,
                                        are_td_errors=True):
    r"""Sampling probabilities and bias-correcting weights.

    .. math::
        P(i) \propto \bigl(|\delta_i| + \varepsilon\bigr)^{\alpha},
        \qquad
        w_i = \frac{(N\,P(i))^{-\beta}}{\max_j w_j}

    Two knobs doing opposite jobs.  :math:`\alpha` decides how much the
    sampling deviates from uniform (``0`` = uniform, ``1`` = fully
    proportional); :math:`\beta` decides how much of the resulting bias
    is corrected back out, and is annealed to 1 by the end of training
    because the bias only matters once the estimates are close.  The
    normalisation by :math:`\max_j w_j` keeps every weight at most 1, so
    prioritisation can only ever *shrink* an update -- otherwise a rare
    sample would blow up the step size.

    Parameters
    ----------
    priorities : array-like
        TD errors (default) or already-formed priorities.
    N : int, optional
        Buffer size; defaults to ``len(priorities)``.
    alpha : float, optional
        Prioritisation exponent in ``[0, 1]``.
    beta : float, optional
        Importance-sampling exponent in ``[0, 1]``.
    eps : float, optional
        Positive floor that keeps zero-error transitions sampleable.
    are_td_errors : bool, optional
        If False, ``priorities`` are used as-is (must be positive).

    Returns
    -------
    RichResult
        Payload keys ``weights``, ``probabilities``, ``max_weight_index``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Prioritized Experience Replay section.

    Examples
    --------
    With ``alpha = 1`` the probabilities are proportional to the errors
    (plus eps), and the largest-probability sample gets the smallest
    weight:

    >>> r = geron_prioritized_experience_weight([3.0, 1.0], alpha=1.0, beta=1.0, eps=0.0)
    >>> [round(p, 6) for p in r["probabilities"]]
    [0.75, 0.25]
    >>> [round(w, 6) for w in r["weights"]]
    [0.333333, 1.0]

    ``alpha = 0`` is uniform sampling, and then every weight is 1:

    >>> geron_prioritized_experience_weight([3.0, 1.0], alpha=0.0, beta=1.0)["weights"]
    [1.0, 1.0]
    """
    d = np.asarray(priorities, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("priorities is empty.")
    if not np.all(np.isfinite(d)):
        raise ValueError("priorities contains non-finite values.")
    alpha = float(alpha)
    beta = float(beta)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must lie in [0, 1], got {beta}.")
    eps = float(eps)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    if are_td_errors:
        p = np.abs(d) + eps
    else:
        if np.any(d <= 0):
            raise ValueError("explicit priorities must be positive.")
        p = d
    if np.all(p == 0):
        raise ValueError("all priorities are zero; raise eps so every transition can be sampled.")
    pa = p**alpha
    P = pa / pa.sum()
    if N is None:
        N = d.size
    N = int(N)
    if N < d.size:
        raise ValueError(f"N={N} is smaller than the {d.size} priorities supplied.")

    w = (N * P) ** (-beta)
    w = w / w.max()

    return RichResult(
        title="Prioritized replay weights",
        summary_lines=[("alpha", alpha), ("beta", beta), ("N", N)],
        payload={
            "weights": w.tolist(),
            "probabilities": P.tolist(),
            "priorities": p.tolist(),
            "max_weight_index": int(np.argmax(w)),
            "estimate": w.tolist(),
            "n": int(d.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpex: P ~ (|delta|+eps)^alpha; w = (N P)^-beta / max w <= 1; alpha biases, beta corrects"
