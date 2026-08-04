# morie.fn -- function file (rootcoder007/morie)
"""The kernel constant r_1 of the KDFE variance."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfr1", "fauzi_r1_integral"]


def kdfr1(kernel="gaussian", lo=-8.0, hi=8.0, ngrid=4001):
    r"""The kernel constant r_1 of the KDFE variance.

    Eq. (2.9):

    .. math:: r_1 = \int_{-\infty}^{\infty} yK(y)W(y)\,dy,

    with :math:`W(y)=\int_{-\infty}^y K(v)\,dv`.

    This is the single constant that makes the whole book worth writing.
    The KDFE variance (2.4) is :math:`F(1-F)/n - 2hr_1f_X(x)/n`: the
    bandwidth enters with a NEGATIVE sign, so smoothing REDUCES variance,
    the opposite of what happens in a density estimator where it enters at
    :math:`1/(nh)`. Sec. 2.1 concludes that any kernel with
    :math:`r_1>0` beats the empirical df for every :math:`F_X`, and the
    book proves :math:`r_1 \ge 0` for symmetric kernels by splitting the
    integral at 0.

    For the Gaussian kernel the value is exactly
    :math:`1/(2\sqrt\pi) \approx 0.2820948`, returned in closed form. Any
    other kernel is integrated on a fixed trapezoid grid -- fixed node
    count, no adaptive refinement, so two calls always agree bitwise.

    Parameters
    ----------
    kernel : {"gaussian"} or callable, default "gaussian"
        ``"gaussian"`` gives the closed form; a callable ``K(y)`` is
        integrated numerically together with its own integral ``W``.
    lo, hi : float, default -8.0, 8.0
        Quadrature limits, used only for a callable kernel.
    ngrid : int, default 4001
        Number of nodes; fixed, never adapted.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``kernel``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (2.4) and (2.9).
    """
    if kernel == "gaussian":
        return RichResult(
            payload={
                "estimate": float(1.0 / (2.0 * np.sqrt(np.pi))),
                "kernel": "gaussian",
                "method": "r_1 = int y K(y) W(y) dy, closed form (Eq. 2.9)",
            }
        )
    if not callable(kernel):
        raise ValueError('kernel must be "gaussian" or a callable K(y).')
    y = np.linspace(float(lo), float(hi), int(ngrid))
    kv = np.asarray([float(kernel(float(t))) for t in y], dtype=float)
    wv = np.asarray(
        [float(np.trapezoid(kv[: i + 1], y[: i + 1])) if i else 0.0 for i in range(y.size)],
        dtype=float,
    )
    val = float(np.trapezoid(y * kv * wv, y))
    return RichResult(
        payload={
            "estimate": val,
            "kernel": "callable",
            "method": "r_1 = int y K(y) W(y) dy, trapezoid (Eq. 2.9)",
        }
    )


fauzi_r1_integral = kdfr1


def cheatsheet():
    return "fzr1: r_1 = int y K W dy > 0 -- why smoothing lowers the KDFE variance (Eq. 2.9)"


# CANONICAL TEST
# >>> abs(kdfr1()['estimate'] - 0.28209479177387814) < 1e-15
# True
