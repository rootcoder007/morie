# morie.fn -- function file (rootcoder007/morie)
"""Huber loss -- Boyd & Vandenberghe Sec. 6.1.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_huber_loss"]


def boyd_huber_loss(u, M=1.0):
    r"""The Huber penalty

    .. math::
        \phi(u) = \begin{cases} u^2 & |u| \le M \\
                                 M(2|u| - M) & |u| > M. \end{cases}

    Quadratic near zero and AFFINE in the tails, joined so that both the
    value and the first derivative are continuous at :math:`\pm M`. The
    two halves are not an approximation of each other -- the affine tail is
    chosen precisely to make the derivative match at the join.

    That bounded derivative is the robustness: under squared error an
    outlier's influence grows without limit, while here it saturates at
    :math:`2M`. A gross outlier can bias a Huber fit, but it cannot
    dominate it the way it dominates least squares.

    Parameters
    ----------
    u : array-like
        Residuals.
    M : float
        Transition point, positive. Small M is closer to :math:`\ell_1`
        and more robust; large M is closer to least squares.

    Returns
    -------
    RichResult
        ``loss``, ``total``, ``gradient``, ``n_outliers`` (residuals in
        the affine region), ``max_influence``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Quadratic inside the transition, affine outside.

    >>> r = boyd_huber_loss([0.5, 1.0, 3.0], M=1.0)
    >>> [float(round(v, 4)) for v in r["loss"]]
    [0.25, 1.0, 5.0]

    Value and derivative are both continuous at the join, which is what
    the M(2|u| - M) form buys.

    >>> lo = boyd_huber_loss([0.999999], M=1.0)
    >>> hi = boyd_huber_loss([1.000001], M=1.0)
    >>> bool(abs(lo["loss"][0] - hi["loss"][0]) < 1e-5
    ...      and abs(lo["gradient"][0] - hi["gradient"][0]) < 1e-5)
    True

    Influence saturates: the gradient never exceeds 2M however large the
    residual.

    >>> float(boyd_huber_loss([1e6], M=1.0)["gradient"][0])
    2.0

    >>> boyd_huber_loss([1.0], M=0.0)
    Traceback (most recent call last):
        ...
    ValueError: M must be positive
    """
    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    m = float(M)
    if m <= 0:
        raise ValueError("M must be positive")
    a = np.abs(uv)
    inner = a <= m
    loss = np.where(inner, uv ** 2, m * (2.0 * a - m))
    grad = np.where(inner, 2.0 * uv, 2.0 * m * np.sign(uv))
    return RichResult(
        title="Huber loss",
        summary_lines=[("n", int(uv.size)), ("M", m),
                       ("total", float(loss.sum())),
                       ("in affine tail", int(np.sum(~inner)))],
        payload={
            "loss": loss, "total": float(loss.sum()), "gradient": grad,
            "quadratic": inner, "n_outliers": int(np.sum(~inner)),
            "max_influence": 2.0 * m, "M": m,
            "method": "boyd_huber_loss",
        },
    )


def cheatsheet():
    return "cvxhrm: influence saturates at 2M -- an outlier can bias a Huber fit but cannot dominate it"
