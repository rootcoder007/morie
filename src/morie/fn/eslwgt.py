# morie.fn -- function file (rootcoder007/morie)
"""Weight decay -- ESL Sec 11.5.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_weight_decay"]


def esl_weight_decay(weights, lambda_=0.01, loss=0.0, norm="l2"):
    r"""Weight-decay penalty, its gradient, and the penalised objective.

    .. math::
        J(w) = L(w) + \lambda \sum_j w_j^2,
        \qquad \nabla_w \big(\lambda \lVert w \rVert_2^2\big) = 2\lambda w .

    The factor of 2 is the part that bites: frameworks that write the penalty
    as :math:`\tfrac{\lambda}{2}\lVert w\rVert^2` have gradient
    :math:`\lambda w`, so the same nominal ``lambda_`` regularises twice as
    hard here. ``gradient`` is returned explicitly so the convention in use is
    visible rather than assumed.

    ESL notes that weight decay is only meaningful on standardised inputs --
    otherwise the penalty is a statement about the units the predictors
    happen to be measured in.

    Parameters
    ----------
    weights : array-like
        Weight vector. Bias terms should be excluded by the caller; they are
        conventionally left unpenalised.
    lambda_ : float
        Penalty strength, non-negative.
    loss : float
        Unpenalised loss, added to give ``objective``.
    norm : {"l2", "l1"}
        ``"l2"`` is classical weight decay (ridge); ``"l1"`` is the lasso
        penalty, whose gradient is a subgradient and is undefined at zero
        (reported as 0, the convention that yields soft thresholding).

    Returns
    -------
    RichResult
        ``penalty``, ``gradient``, ``objective``, ``effective_lambda``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    >>> r = esl_weight_decay([3.0, -4.0], lambda_=0.5)
    >>> float(r["penalty"])
    12.5
    >>> [float(v) for v in r["gradient"]]
    [3.0, -4.0]

    The L1 subgradient is the sign, taken as 0 at exactly zero.

    >>> [float(v) for v in esl_weight_decay([2.0, 0.0, -1.0], lambda_=1.0, norm="l1")["gradient"]]
    [1.0, 0.0, -1.0]

    >>> esl_weight_decay([1.0], lambda_=-1.0)
    Traceback (most recent call last):
        ...
    ValueError: lambda_ must be non-negative
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    w = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
    if norm == "l2":
        pen = float(lambda_ * np.sum(w**2))
        grad = 2.0 * lambda_ * w
    elif norm == "l1":
        pen = float(lambda_ * np.sum(np.abs(w)))
        grad = lambda_ * np.sign(w)
    else:
        raise ValueError('norm must be "l2" or "l1"')
    return RichResult(
        title=f"Weight decay ({norm})",
        summary_lines=[("lambda", float(lambda_)), ("penalty", pen)],
        payload={
            "penalty": pen, "gradient": grad,
            "objective": float(loss) + pen,
            "effective_lambda": float(2 * lambda_) if norm == "l2" else float(lambda_),
            "norm": norm, "n_weights": int(w.size),
            "method": "esl_weight_decay",
        },
    )


def cheatsheet():
    return "eslwgt: penalty lambda*||w||^2 so the gradient is 2*lambda*w -- twice the lambda/2 convention"
