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

from . import _array_core as np

__all__ = ["kernel_K", "kernel_W", "kernel_V", "mu2", "gamma_kernel_density",
           "boundary_free_transform", "muller_order_m",
           "rratio", "agamma_kernel"]


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
    from . import _stats_core as stats
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
    from . import _stats_core as stats

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


def kdfe_bandwidth(x, sigma=None, n=None):
    r"""Normal-reference bandwidth for a DISTRIBUTION-function-type
    kernel estimator: :math:`(4)^{1/3}\,\hat\sigma\,n^{-1/3}`.

    A cube root, not the fifth root of the density rule, and the
    difference is not cosmetic. Equations (2.3)-(2.4) of the book give

    .. math::
        \mathrm{Bias}[\hat F_h(x)] &= h^2 \tfrac{f_X'(x)}2 \mu_2(K)
          + o(h^2), \\
        \mathrm{Var}[\hat F_h(x)] &= \tfrac1n F(1-F)
          - \tfrac{2h}{n} r_1 f_X(x) + o(h/n),

    with :math:`r_1 = \int y K(y) W(y) dy`, so the MISE is

    .. math:: \frac{h^4}4 \mu_2^2 R(f') + \frac1n\!\int\! F(1-F)
              - \frac{2h}n r_1 .

    The bandwidth enters the variance at order :math:`h/n` and with a
    NEGATIVE sign -- smoothing *reduces* variance here, where in a
    density estimator it enters at :math:`1/(nh)` and blows up as
    :math:`h \to 0`. Setting the derivative to zero gives

    .. math:: h_{opt} = \Big(\frac{2 r_1}{n\,\mu_2^2\,R(f')}\Big)^{1/3},

    and for a Gaussian kernel (:math:`\mu_2 = 1`,
    :math:`r_1 = 1/(2\sqrt\pi)`) against a normal reference
    (:math:`R(f') = 1/(4\sqrt\pi\sigma^3)`) that collapses to
    :math:`(4)^{1/3}\sigma n^{-1/3} \approx 1.587\,\sigma\,n^{-1/3}`.

    Sec. 5.3.2 states the same conclusion in words -- "Azzalini in
    [9] recommended a bandwidth of :math:`cn^{-1/3}` for the
    estimation of the distribution function" -- and the book's own
    simulations use :math:`h_n = n^{-1/3}`.

    Everything in this book that converges at the parametric rate
    with an :math:`O(h^2)` bias and an :math:`O(h/n)` variance term
    takes this rate: the KDFE itself, the smoothed KS, CvM, sign and
    Wilcoxon statistics, the survival estimator and the mean residual
    life estimators (whose Theorem 4.3 variance is likewise
    :math:`O(1/n) - O(h/n)`). Only the Ch. 1 DENSITY estimators,
    whose variance is :math:`O(1/(nh))`, take :math:`n^{-1/5}`.

    Under the density rule a KDFE oversmooths badly enough to lose,
    in mean squared error, to the empirical distribution function it
    exists to improve on.

    Parameters
    ----------
    x : array-like, optional
        Sample; ``sigma`` and ``n`` are read off it when not given.
    sigma : float, optional
        Scale estimate, overriding the one taken from ``x``.
    n : int, optional
        Sample size, overriding ``len(x)``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (2.3), (2.4) and the MISE display
    of Sec. 2.1; Sec. 5.3.2. Azzalini, A. (1981), *Biometrika*
    68:326-328.
    """
    if x is not None:
        xv = np.asarray(x, dtype=float).ravel()
        if n is None:
            n = xv.size
        if sigma is None:
            s = float(np.std(xv, ddof=1))
            iqr = float(np.subtract(*np.percentile(xv, [75, 25]))) / 1.349
            sigma = min(s, iqr) if iqr > 0 else s
    if n is None or n < 2:
        raise ValueError("need a sample size of at least 2.")
    if sigma is None or sigma <= 0:
        sigma = 1.0
    return float(4.0 ** (1.0 / 3.0) * sigma * n ** (-1.0 / 3.0))


def rratio(z):
    r"""The book's :math:`R(z) = \sqrt{2\pi}\,z^{z+1/2}/(e^{z}\Gamma(z+1))`,
    Eq. (1.12).

    It is the Stirling defect of the gamma function: ``R(z)`` increases
    monotonically to 1 from below (Remark 1.2), which is exactly the fact
    the book uses to read off the O(n^-1 h^-1/4) interior and
    O(n^-1 h^-3/4) boundary rates of Var[A_h(x)] in (1.11).

    Computed through log-gamma so that the ``z^(z+1/2)`` factor does not
    overflow for the small bandwidths this suite uses (``z ~ h^{-1/2}``).
    """
    import math as _math
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("R(z) is defined for z > 0.")
    log_r = 0.5 * np.log(2.0 * np.pi) + (z + 0.5) * np.log(z) - z - np.asarray([_math.lgamma(float(v) + 1.0) for v in np.atleast_1d(z)]).reshape(np.shape(z)) if np.ndim(z) else _math.lgamma(float(z) + 1.0)
    return np.exp(log_r)


def agamma_kernel(x, v, h):
    r"""``A_h(v)`` of Eq. (1.9): the mean over the sample of the
    Gamma(:math:`h^{-1/2}`, :math:`v\sqrt h + h`) density.

    This is NOT Chen's gamma kernel of ``gamma_kernel_density`` -- Chen
    takes shape ``v/h + 1`` and scale ``h``, whose shape moves with the
    evaluation point; the book instead fixes the shape at
    :math:`h^{-1/2}` and moves the SCALE. That single change is what buys
    the smaller variance orders of Remark 1.2, and it is also what costs
    the bias its rate, taking it from O(h) up to O(sqrt h) in (1.10) --
    which is then bought back by the geometric extrapolation (1.14).
    """
    from . import _stats_core as stats
    xv = np.asarray(x, dtype=float).ravel()
    hh = float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    if np.any(xv < 0):
        raise ValueError("gamma kernels need data on [0, infinity).")
    shape = 1.0 / np.sqrt(hh)
    g = np.atleast_1d(np.asarray(v, dtype=float))
    if np.any(g < 0):
        raise ValueError("the evaluation points must lie in [0, infinity).")
    out = np.empty(g.size)
    for i, pt in enumerate(g):
        scale = float(pt) * np.sqrt(hh) + hh
        out[i] = float(np.mean(stats.gamma.pdf(xv, a=shape, scale=scale)))
    return out
