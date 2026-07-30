# morie.fn -- function file (rootcoder007/morie)
"""Sparse autoencoder activation penalty."""

import numpy as np

from ._richresult import RichResult

__all__ = ["sparse_autoencoder_penalty", "geron_sparse_autoencoder"]


def sparse_autoencoder_penalty(activations, target=0.1, weight=1.0,
                               kind="kl"):
    r"""Sparsity penalty on the mean activation of a coding layer.

    With :math:`\hat\rho_j` the mean activation of unit :math:`j` over
    the batch and :math:`\rho` the target, the KL penalty is

    .. math::
       \sum_j \rho\log\frac{\rho}{\hat\rho_j}
            + (1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}.

    The KL form is preferred over an :math:`\ell_1` penalty on
    :math:`\hat\rho` for a specific reason: its gradient blows up as
    :math:`\hat\rho_j` approaches 0 or 1, so it pushes hard away from
    the degenerate solutions where a unit is always off (dead, and
    never recoverable) or always on (carrying no information). An
    :math:`\ell_1` penalty has constant gradient and will happily kill
    units outright.

    ``dead_units`` and ``saturated_units`` count how many are already
    at those boundaries, which is what the penalty exists to prevent
    and the first thing to look at when a sparse autoencoder trains to
    a useless code.

    Parameters
    ----------
    activations : array-like, shape (n, k)
        Coding-layer activations, in [0, 1].
    target : float
        Target mean activation.
    weight : float
        Penalty multiplier.
    kind : {'kl', 'l1'}

    Returns
    -------
    RichResult
        ``penalty``, ``per_unit``, ``mean_activation``, ``dead_units``,
        ``saturated_units``, ``achieved_sparsity``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 17,
    sparse autoencoders. Ng (2011), CS294A lecture notes.

    Examples
    --------
    >>> out = sparse_autoencoder_penalty([[0.1, 0.1], [0.1, 0.1]],
    ...                                  target=0.1)
    >>> round(float(out["penalty"]), 10)
    0.0
    """
    A = np.atleast_2d(np.asarray(activations, dtype=float))
    if A.ndim != 2:
        raise ValueError("activations must be 2-dimensional.")
    if np.any(A < -1e-9) or np.any(A > 1 + 1e-9):
        raise ValueError(
            "activations must lie in [0, 1]; apply a sigmoid first."
        )
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1), got %r." % target)
    if kind not in ("kl", "l1"):
        raise ValueError("kind must be 'kl' or 'l1', got %r." % kind)

    rho_hat = A.mean(axis=0)
    eps = 1e-12
    r = np.clip(rho_hat, eps, 1 - eps)
    if kind == "kl":
        per = (target * np.log(target / r)
               + (1 - target) * np.log((1 - target) / (1 - r)))
    else:
        per = np.abs(r - target)
    pen = float(weight * np.sum(per))
    return RichResult(
        payload={
            "estimate": pen,
            "penalty": pen,
            "per_unit": per,
            "mean_activation": rho_hat,
            "target": float(target),
            "achieved_sparsity": float(np.mean(rho_hat)),
            "dead_units": int(np.sum(rho_hat < 1e-6)),
            "saturated_units": int(np.sum(rho_hat > 1 - 1e-6)),
            "boundary_note": (
                "the KL gradient diverges as the mean activation nears 0 or "
                "1, which is exactly what keeps units off those boundaries; "
                "an l1 penalty has constant gradient and will kill units"
            ),
            "kind": kind,
            "weight": float(weight),
            "n_units": int(A.shape[1]),
            "n": int(A.shape[0]),
            "method": "Sparse-autoencoder %s activation penalty" % kind.upper(),
        }
    )


def cheatsheet():
    return (
        "hmspae: KL (or l1) sparsity penalty on mean activation, with dead "
        "and saturated unit counts"
    )


#: Catalogue alias for :func:`sparse_autoencoder_penalty`.
geron_sparse_autoencoder = sparse_autoencoder_penalty
