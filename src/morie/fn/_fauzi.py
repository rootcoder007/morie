# morie.fn -- internal helpers (rootcoder007/morie)
"""Kernel distribution-function estimation.

Spec: Fauzi, R. R. and Maesono, Y., *Statistical Inference Based on
Kernel Distribution Function Estimators*, SpringerBriefs in
Statistics, 2023. Equation numbers below are the book's and were
checked against the text.

The book's organising problem is the BOUNDARY. A symmetric kernel
placed near the edge of a bounded support puts mass outside it, and
the resulting bias does not vanish as ``h -> 0`` -- it is O(h) at
the boundary against O(h^2) in the interior. Every construction here
is a way around that: gamma kernels whose support matches the data
(Ch. 1), and bijective transformations that move a bounded problem to
the whole line and back (Ch. 4).
"""

import numpy as np

__all__ = ["kernel_K", "kernel_W", "kernel_V", "mu2", "gamma_kernel_density",
           "boundary_free_transform", "muller_order_m"]


def kernel_K(u):
    """Gaussian kernel."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def kernel_W(u):
    r"""``W(u) = int_{-inf}^{u} K(v)dv``, the INTEGRATED kernel used
    by the distribution-function estimator (2.2).

    A distribution function estimator smooths with the kernel's
    integral, not the kernel itself. That is what makes it continuous
    where the empirical df jumps, and it is why the bias carries
    :math:`h^2\\mu_2(K)f'(x)/2` rather than the density estimator's
    :math:`f''`.
    """
    from scipy import stats
    return stats.norm.cdf(np.asarray(u, dtype=float))


def kernel_V(u):
    r"""``V(u) = 1 - W(u)``, the survival counterpart used in the
    mean-residual-life estimators of Ch. 4."""
    return 1.0 - kernel_W(u)


def mu2(kernel="gaussian"):
    r""":math:`\\mu_2(K) = \\int u^2 K(u)du`, the constant in every
    bias expansion in the book. One for the Gaussian kernel."""
    if kernel == "gaussian":
        return 1.0
    if kernel == "epanechnikov":
        return 0.2
    raise ValueError("kernel must be 'gaussian' or 'epanechnikov'.")


def gamma_kernel_density(x, grid, h):
    r"""Chen's (1999) gamma kernel density estimator for data on
    :math:`[0,\\infty)`:

    .. math:: \\hat f(x) = \\frac1n\\sum_i
              K_{x/h + 1,\\,h}(X_i),

    with :math:`K_{a,b}` the Gamma(a, b) density.

    The kernel's SHAPE changes with the evaluation point, and its
    support is exactly :math:`[0,\\infty)`. No mass ever lands on the
    negative half-line, so the boundary bias that afflicts a
    symmetric kernel simply does not arise -- the estimator is
    consistent at zero, where a Gaussian-kernel estimate is not.
    """
    from scipy import stats

    xv = np.asarray(x, dtype=float).ravel()
    g = np.atleast_1d(np.asarray(grid, dtype=float))
    hh = float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    if np.any(xv < 0):
        raise ValueError("gamma kernels need data on [0, infinity).")
    if np.any(g < 0):
        raise ValueError("the grid must lie in [0, infinity).")
    out = np.empty(g.size)
    for i, v in enumerate(g):
        shape = v / hh + 1.0
        out[i] = float(np.mean(stats.gamma.pdf(xv, a=shape, scale=hh)))
    return out


def boundary_free_transform(kind="log"):
    r"""A bijection :math:`g` from the whole line onto the support,
    with its inverse and first two derivatives (Ch. 4).

    The Ch. 4 construction estimates on the TRANSFORMED scale, where
    a symmetric kernel is legitimate because the support is
    unbounded, then maps back. The derivatives of ``g`` are what
    appear in the bias coefficients ``b_1``, ``b_2`` and ``b_3``
    (4.14, 4.15, 4.21) -- the transformation does not remove the bias,
    it makes it computable and O(h^2) everywhere including at the
    edge.
    """
    if kind == "log":
        return {"g": np.exp, "g_inv": np.log,
                "dg": np.exp, "d2g": np.exp,
                "support": (0.0, np.inf), "name": "exp/log"}
    if kind == "identity":
        one = np.ones_like
        return {"g": lambda z: z, "g_inv": lambda t: t,
                "dg": one, "d2g": lambda z: np.zeros_like(z),
                "support": (-np.inf, np.inf), "name": "identity"}
    raise ValueError("kind must be 'log' or 'identity'.")


def muller_order_m(u, m=4):
    r"""Order-m kernel (Muller 1991): satisfies
    :math:`\\int u^j K = 0` for :math:`1 \\le j < m` and finite at
    :math:`j = m`.

    Higher order buys a faster bias rate, :math:`O(h^m)` instead of
    :math:`O(h^2)`, and pays for it by taking NEGATIVE values -- so
    the resulting density estimate can be negative, and the
    distribution estimate non-monotone. That is the trade the book
    makes explicit, and it is why order-m kernels are used for
    quantiles, where the estimand is a location rather than a
    density.
    """
    u = np.asarray(u, dtype=float)
    m = int(m)
    if m == 2:
        return kernel_K(u)
    if m == 4:
        return (3.0 - u ** 2) / 2.0 * kernel_K(u)
    if m == 6:
        return (15.0 - 10.0 * u ** 2 + u ** 4) / 8.0 * kernel_K(u)
    raise ValueError("m must be 2, 4 or 6.")


def cheatsheet():
    return "_fauzi: the boundary is the problem -- gamma kernels and bijections are the two answers"
