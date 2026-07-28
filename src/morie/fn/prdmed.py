# morie.fn -- function file (rootcoder007/morie)
"""Product-of-coefficients mediation estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["product_of_coefficients"]


def product_of_coefficients(a, b, se_a=None, se_b=None, n=None,
                            alpha=0.05, n_boot=0, seed=0):
    r"""Indirect effect as :math:`a\,b`, with an honest interval.

    In the two-equation linear system

    .. math::
       M = i_1 + aX + e_1, \qquad Y = i_2 + c'X + bM + e_2,

    the natural indirect effect is :math:`ab`.

    The interval is where this goes wrong in practice. The Sobel
    standard error, :math:`\sqrt{a^2s_b^2 + b^2s_a^2}`, assumes
    :math:`ab` is normally distributed. It is not: the product of two
    normals is heavy-tailed and ASYMMETRIC, so a symmetric interval
    misses on one side, and the error is worst exactly where mediation
    studies live -- small-to-moderate effects. The bootstrap percentile
    interval respects that asymmetry and is what MacKinnon's
    simulations recommend. Both are returned; ``sobel_symmetric`` marks
    which is which.

    Note also that :math:`ab` earns its causal reading only under
    no-unmeasured-confounding of the MEDIATOR-outcome relation, which
    randomising :math:`X` does not deliver. Randomisation fixes
    :math:`a`; nothing about it makes :math:`b` causal.

    Parameters
    ----------
    a, b : float or array-like
        Path coefficients, or bootstrap draws of them.
    se_a, se_b : float, optional
        Enables the Sobel interval.
    n : int, optional
    alpha : float
    n_boot : int
        Draws for a parametric bootstrap when SEs are supplied.
    seed : int

    Returns
    -------
    RichResult
        ``indirect``, ``sobel_se``, ``sobel_ci``, ``boot_ci``,
        ``skewness``, ``sobel_symmetric``.

    References
    ----------
    MacKinnon (2008), *Introduction to Statistical Mediation
    Analysis*, Erlbaum, chapters 3-4.
    Sobel (1982), *Sociological Methodology* 13:290-312.
    MacKinnon, Lockwood and Williams (2004), *Multivariate Behavioral
    Research* 39:99-128, on the asymmetry.

    Examples
    --------
    >>> float(product_of_coefficients(0.5, 0.4)["indirect"])
    0.2
    """
    av = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if av.size != bv.size and av.size != 1 and bv.size != 1:
        raise ValueError(
            "a and b must be scalars or arrays of the same length, got "
            "%d and %d." % (av.size, bv.size)
        )
    ab = av * bv
    point = float(np.mean(ab))
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z(1 - alpha / 2)

    sobel_se = sobel_ci = None
    if se_a is not None and se_b is not None:
        sa, sb = float(se_a), float(se_b)
        if sa < 0 or sb < 0:
            raise ValueError("standard errors must be non-negative.")
        a0, b0 = float(av[0]), float(bv[0])
        sobel_se = float(np.sqrt(a0 ** 2 * sb ** 2 + b0 ** 2 * sa ** 2))
        sobel_ci = (point - z * sobel_se, point + z * sobel_se)

    boot_ci = None
    skew = np.nan
    draws = None
    if av.size > 1:
        draws = ab
    elif n_boot and se_a is not None and se_b is not None:
        rng = np.random.default_rng(int(seed))
        draws = (rng.normal(float(av[0]), float(se_a), int(n_boot))
                 * rng.normal(float(bv[0]), float(se_b), int(n_boot)))
    if draws is not None and draws.size > 2:
        lo = float(np.quantile(draws, alpha / 2))
        hi = float(np.quantile(draws, 1 - alpha / 2))
        boot_ci = (lo, hi)
        c = draws - draws.mean()
        s = float(draws.std(ddof=1))
        skew = float(np.mean(c ** 3) / s ** 3) if s > 0 else np.nan

    return RichResult(
        payload={
            "estimate": point,
            "indirect": point,
            "a": float(av[0]) if av.size == 1 else av,
            "b": float(bv[0]) if bv.size == 1 else bv,
            "sobel_se": sobel_se,
            "sobel_ci": sobel_ci,
            "sobel_symmetric": True if sobel_ci is not None else None,
            "sobel_note": (
                "the Sobel interval assumes ab is normal; the product of two "
                "normals is heavy-tailed and asymmetric, so it misses on one "
                "side, worst at the small-to-moderate effects mediation "
                "studies actually report"
            ),
            "boot_ci": boot_ci,
            "boot_note": (
                None if boot_ci is None else
                "percentile interval, which respects the asymmetry the Sobel "
                "interval cannot"
            ),
            "skewness": skew,
            "asymmetry": (None if boot_ci is None or sobel_ci is None else
                          float(abs((boot_ci[1] - point)
                                    - (point - boot_ci[0])))),
            "identification_note": (
                "ab is causal only under no unmeasured confounding of the "
                "MEDIATOR-outcome relation; randomising X fixes a and does "
                "nothing for b"
            ),
            "n": None if n is None else int(n),
            "method": "Product-of-coefficients indirect effect",
        }
    )


def _z(qq):
    import math
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < qq:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "prdmed: indirect effect ab with Sobel and bootstrap intervals, and "
        "why the symmetric one is wrong"
    )
