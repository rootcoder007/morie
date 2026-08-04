# morie.fn -- function file (rootcoder007/morie)
"""Edgeworth expansion function G_n for the kernel quantile estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["qedgew", "fauzi_gn_edgeworth_correction"]


def qedgew(x, n, h, sigma, e1, e2, e3, e4, e5, e6, delta=0.0, book=False):
    r"""Edgeworth expansion function G_n for the kernel quantile estimator.

    Theorem 3.1, the function :math:`G_n` of (3.14):

    .. math::
        G_n(x) = \Phi(x) - \phi(x)\Big\{
        \frac{x^2-1}{6n^{1/2}\sigma_n^3}\Big(e_{1n}+\frac{3e_{2n}}h\Big)
        + \frac1{nh^2}\Big[\frac x{4\sigma_n^2}(4e_{5n}+e_{6n})
        + \frac{x^3-3x}{6\sigma_n^4}(3e_{3n}+e_{4n})
        + \frac{x^5-10x^3+15x}{8\sigma_n^6}e_{2n}^2\Big]\Big\}.

    Every bracket is a Hermite polynomial -- :math:`He_2 = x^2-1`,
    :math:`He_3 = x^3-3x`, :math:`He_5 = x^5-10x^3+15x` -- and the
    :math:`He_5` term carries :math:`e_{2n}` SQUARED, which is the
    standard Edgeworth structure in which the fifth-order term is the
    square of the third-order one.

    The book's (3.14) prints :math:`\{3e_{2n}+e_{4n}\}` in the
    :math:`He_3` bracket. The primary source -- Maesono, Y. and Penev, S.
    (2011), "Edgeworth expansion for the kernel quantile estimator",
    *Annals of the Institute of Statistical Mathematics* 63(3):617-644,
    Theorem 1 -- prints :math:`\{3e_{3n}+e_{4n}\}`. The paper is followed
    by default. The tell that the book has a typo is internal: it defines
    :math:`e_{3n}` immediately below the display and then never uses it
    anywhere. Pass ``book=True`` to reproduce the book's spelling.

    The six :math:`e_{jn}` are the moment functionals defined below the
    display; they depend on the kernel, the bandwidth and :math:`Q'`, so
    they are the caller's to supply.

    Parameters
    ----------
    x : float or array-like
        Argument of the expansion.
    n : int
        Sample size.
    h : float
        Bandwidth.
    sigma : float
        ``sigma_n``, the standardising scale.
    e1, e2, e3, e4, e5, e6 : float
        The moment functionals ``e_{1n}`` ... ``e_{6n}``.
    delta : float, default 0.0
        The shift ``delta / (sigma sqrt(n))`` of (3.14); 0 gives
        ``G_n(x)`` itself.
    book : bool, default False
        Use the book's ``3 e_{2n} + e_{4n}`` instead of the primary
        source's ``3 e_{3n} + e_{4n}``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``normal``, ``correction``, ``book``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 3.1, Eq. (3.14); Maesono and Penev (2011), AISM 63:617-644, Theorem 1.
    """
    from . import _stats_core as stats

    n = int(n)
    h = float(h)
    s = float(sigma)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if s <= 0:
        raise ValueError(f"sigma_n must be positive, got {s}.")
    xv = np.atleast_1d(np.asarray(x, dtype=float)) - float(delta) / (s * np.sqrt(n))
    phi = stats.norm.pdf(xv)
    base = stats.norm.cdf(xv)
    he2 = xv ** 2 - 1.0
    he3 = xv ** 3 - 3.0 * xv
    he5 = xv ** 5 - 10.0 * xv ** 3 + 15.0 * xv
    term1 = he2 / (6.0 * np.sqrt(n) * s ** 3) * (float(e1) + 3.0 * float(e2) / h)
    mid = float(e2) if book else float(e3)
    inner = (
        xv / (4.0 * s ** 2) * (4.0 * float(e5) + float(e6))
        + he3 / (6.0 * s ** 4) * (3.0 * mid + float(e4))
        + he5 / (8.0 * s ** 6) * float(e2) ** 2
    )
    corr = phi * (term1 + inner / (n * h * h))
    return RichResult(
        payload={
            "estimate": [float(v) for v in (base - corr)],
            "normal": [float(v) for v in base],
            "correction": [float(v) for v in corr],
            "book": bool(book),
            "method": "Edgeworth function G_n for the kernel quantile estimator (3.14)",
        }
    )


fauzi_gn_edgeworth_correction = qedgew


def cheatsheet():
    return "fzgn: G_n: Hermite He2, He3, He5 with e2n SQUARED; the book's 3e2n should be 3e3n"


# CANONICAL TEST
# >>> r = qedgew(x=0.0, n=100, h=0.1, sigma=1.0, e1=0.0, e2=0.0, e3=0.0, e4=0.0, e5=0.0, e6=0.0)
# >>> abs(r['estimate'][0] - 0.5) < 1e-15
# True
