# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-loss (cross-entropy) cost for binary logistic regression."""

from . import _array_core as np

from ._richresult import RichResult
from .grlogp import geron_logistic_regression_probability

__all__ = ["geron_logistic_cross_entropy_cost"]

_METHOD = "Binary logistic log-loss (Eq 4-17)"


def geron_logistic_cross_entropy_cost(X, y, theta, eps=1e-15):
    r"""Géron Eq 4-17.

    .. math::
        J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Bigl[
        y^{(i)} \log \hat p^{(i)}
        + (1-y^{(i)}) \log(1 - \hat p^{(i)})\Bigr]

    Probabilities come from
    :func:`morie.fn.grlogp.geron_logistic_regression_probability`.
    A saturated probability paired with the opposite label makes the
    exact log-loss infinite; rather than return ``inf`` or ``nan``
    silently, the probabilities are clipped into
    ``[eps, 1 - eps]`` and the number of clipped instances is reported
    both in ``n_clipped`` and as a warning on the result.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Labels, each 0 or 1.
    theta : array-like, shape (n,)
    eps : float, optional
        Clipping floor, default ``1e-15``.

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``probabilities``, ``per_instance_loss``,
        ``n_clipped``, ``accuracy``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-17 (Logistic Regression cost).

    Examples
    --------
    A model that predicts 0.5 for everything pays ``log 2`` per
    instance, whatever the labels:

    >>> X = [[1.0], [1.0]]
    >>> r = geron_logistic_cross_entropy_cost(X, [1.0, 0.0], [0.0])
    >>> round(r["cost"], 10)
    0.6931471806

    Confidence in the right direction is cheaper than the coin flip;
    the same confidence in the wrong direction is dearer:

    >>> right = geron_logistic_cross_entropy_cost([[2.0]], [1.0], [1.0])["cost"]
    >>> wrong = geron_logistic_cross_entropy_cost([[2.0]], [0.0], [1.0])["cost"]
    >>> round(right, 9), round(wrong, 9)
    (0.126928011, 2.126928011)
    """
    probs = geron_logistic_regression_probability(X, theta)
    p = np.asarray(probs["probability"], dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError(f"y has {y.size} entries but X has {p.size} rows.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y must contain only 0 and 1 for binary log-loss.")
    eps = float(eps)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}.")

    n_clipped = int(np.sum((p < eps) | (p > 1.0 - eps)))
    pc = np.clip(p, eps, 1.0 - eps)
    per = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    cost = float(np.mean(per))
    acc = float(np.mean((p >= 0.5).astype(float) == y))

    warns = []
    if n_clipped:
        warns.append(
            f"{n_clipped} of {p.size} predicted probabilities were saturated and "
            f"clipped to [{eps}, {1 - eps}]; the reported cost is a finite lower "
            f"bound on the exact log-loss."
        )

    return RichResult(
        title="Logistic regression log-loss",
        summary_lines=[("Cost", cost), ("Accuracy", acc), ("m", int(p.size))],
        warnings=warns,
        payload={
            "cost": cost,
            "probabilities": p.tolist(),
            "per_instance_loss": per.tolist(),
            "n_clipped": n_clipped,
            "accuracy": acc,
            "estimate": cost,
            "n": int(p.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlogc: J = -mean(y log p + (1-y) log(1-p)) -- Geron Eq 4-17"
