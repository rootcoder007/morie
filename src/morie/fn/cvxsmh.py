# morie.fn -- function file (rootcoder007/morie)
"""Smoothed Huber gradient -- Boyd & Vandenberghe Sec. 6.1.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_smooth_huber_grad"]


def boyd_smooth_huber_grad(u, M=1.0):
    r"""The derivative of the (unit-scaled) Huber penalty,

    .. math::
        \phi'(u) = \begin{cases} u/M & |u| \le M \\
                                  \operatorname{sign}(u) & |u| > M,\end{cases}

    which is the clipped identity, and the reason Huber is called a
    smoothed :math:`\ell_1`: this is exactly :math:`\operatorname{sign}(u)`
    with the discontinuity at zero replaced by a linear ramp of width
    :math:`2M`.

    The derivative is CONTINUOUS but not differentiable at
    :math:`\pm M` -- the second derivative jumps from :math:`1/M` to 0 --
    so Newton's method sees a discontinuous Hessian there and can chatter.
    That is the practical cost of the smoothing being only first order.

    Note the scaling: this is the gradient of the M-normalised Huber
    (:math:`u^2/2M` inside), not of the :math:`u^2` form in
    :func:`~morie.fn.cvxhrm.boyd_huber_loss`. Mixing the two conventions
    is the standard way to get a factor of 2M wrong.

    Parameters
    ----------
    u : array-like
        Residuals.
    M : float
        Transition width, positive.

    Returns
    -------
    RichResult
        ``gradient``, ``saturated`` (entries at :math:`\pm 1`),
        ``second_derivative``, ``n_saturated``, ``approximates``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Linear ramp inside, clipped outside.

    >>> r = boyd_smooth_huber_grad([-2.0, -0.5, 0.0, 0.5, 2.0], M=1.0)
    >>> [float(v) for v in r["gradient"]]
    [-1.0, -0.5, 0.0, 0.5, 1.0]

    Bounded by 1 in magnitude however large the residual -- the clipped
    identity.

    >>> float(boyd_smooth_huber_grad([1e9])["gradient"][0])
    1.0

    Continuous at the join, but the SECOND derivative jumps there, which
    is what makes Newton chatter at the kink.

    >>> lo = boyd_smooth_huber_grad([0.999999])
    >>> hi = boyd_smooth_huber_grad([1.000001])
    >>> bool(abs(lo["gradient"][0] - hi["gradient"][0]) < 1e-5)
    True
    >>> float(lo["second_derivative"][0]), float(hi["second_derivative"][0])
    (1.0, 0.0)

    As M shrinks it approaches sign(u), the l1 subgradient.

    >>> float(boyd_smooth_huber_grad([0.1], M=1e-6)["gradient"][0])
    1.0
    """
    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    m = float(M)
    if m <= 0:
        raise ValueError("M must be positive")
    a = np.abs(uv)
    inner = a <= m
    grad = np.where(inner, uv / m, np.sign(uv))
    d2 = np.where(inner, 1.0 / m, 0.0)
    sat = ~inner
    return RichResult(
        title="Smoothed Huber gradient",
        summary_lines=[("n", int(uv.size)), ("M", m),
                       ("saturated", int(sat.sum()))],
        payload={
            "gradient": grad, "saturated": sat,
            "second_derivative": d2, "n_saturated": int(sat.sum()),
            "approximates": "sign(u), with a ramp of width 2M at the origin",
            "M": m, "method": "boyd_smooth_huber_grad",
        },
    )


def cheatsheet():
    return "cvxsmh: clipped identity = smoothed sign; the SECOND derivative jumps at +/-M"
