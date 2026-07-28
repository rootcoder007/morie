# morie.fn -- function file (rootcoder007/morie)
"""Order-m kernel condition for quantile estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_order_m_kernel"]


def fauzi_order_m_kernel(u, m=4):
    r"""Order-m kernel condition (Fauzi Ch. 3):

    .. math:: \int u^j K(u)du =
              \begin{cases}1 & j = 0\\ 0 & 1 \le j < m\\
              \text{finite, nonzero} & j = m\end{cases}

    Killing the low-order moments pushes the leading bias term from
    :math:`O(h^2)` out to :math:`O(h^m)`, which is how a
    higher-order kernel buys a faster rate without more data.

    The cost is unavoidable and is the reason these kernels are not
    used everywhere: a kernel with vanishing second moment must take
    NEGATIVE values. A density estimate built from one can be
    negative, and a distribution estimate can fail to be monotone.
    That is tolerable for a quantile, where the estimand is a
    location and the smoothing happens in the probability argument,
    and it is not tolerable for a density one intends to plot. The
    module returns the realised moments and the negativity flag.

    Parameters
    ----------
    u : array-like
        Evaluation points.
    m : {2, 4, 6}
        Kernel order.

    Returns
    -------
    RichResult
        keys: ``u``, ``K``, ``order``, ``moments``,
        ``takes_negative_values``, ``bias_order``, ``tradeoff``,
        ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Ch. 3; Muller (1991).
    """
    from ._fauzi import muller_order_m

    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    mm = int(m)
    K = muller_order_m(uv, mm)
    grid = np.linspace(-10, 10, 4001)
    Kg = muller_order_m(grid, mm)
    moments = {j: float(np.trapezoid(grid ** j * Kg, grid))
               for j in range(0, mm + 1)}
    return RichResult(payload={
        "u": uv, "K": K, "order": mm, "moments": moments,
        "takes_negative_values": bool(np.any(Kg < -1e-12)),
        "bias_order": f"O(h^{mm})",
        "tradeoff": "a vanishing second moment forces negative values, so the "
                    "density can go negative and the distribution "
                    "non-monotone; acceptable for quantiles, not for a "
                    "density to be plotted",
        "method": f"Order-{mm} kernel; moments 1..{mm - 1} vanish, pushing the bias to O(h^{mm})"})


def cheatsheet():
    return "fzkoc: order-m kernels MUST go negative -- fine for quantiles, not for a plotted density"
