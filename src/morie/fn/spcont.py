"""Mean-square continuity condition for random fields."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_spatial_continuity"]


def schabenberger_spatial_continuity(cov_func, tol=1e-8):
    r"""
    Mean-square continuity, decided by the covariance at the origin.

    For a field with constant mean and variance, MS continuity at s means

    .. math::

        \lim_{h \to 0} E\!\left[(Z(s) - Z(s+h))^2\right] = 0

    and since

    .. math::

        E\!\left[(Z(s) - Z(s+h))^2\right] = 2\mathrm{Var}[Z(s)] - 2C(h)
        = 2(C(0) - C(h)) = 2\gamma(h)

    the field is mean-square continuous IF AND ONLY IF :math:`C` is
    continuous at the origin. The whole question reduces to the behaviour
    of the covariance function near zero.

    The practical consequence the book draws: a process with a NUGGET
    EFFECT has a discontinuity at the origin and therefore cannot be
    mean-square continuous. The gap reported here is that nugget.

    Parameters
    ----------
    cov_func : callable
        ``C(h)``, taking an array of lags and returning an array.
    tol : float, default 1e-8
        Gap below which the limit is treated as reaching ``C(0)``.

    Returns
    -------
    RichResult
        ``is_continuous``, ``c0``, ``limit_at_zero_plus``, ``gap``
        (the nugget), ``gamma_limit`` (:math:`2\times` the gap, the limit
        of the mean squared difference).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.3, pp. 49-50.
    """
    if not callable(cov_func):
        raise TypeError("`cov_func` must be callable, C(h) -> array")
    c0 = float(np.asarray(cov_func(np.zeros(1))).ravel()[0])
    hs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    approach = np.asarray(cov_func(hs), dtype=float).ravel()
    gaps = c0 - approach

    # The decision is whether the gap TENDS TO ZERO, not whether it is
    # below some fixed number. A continuous C has gaps shrinking with h;
    # a nugget leaves them on a plateau at c0 - the continuous part. A
    # fixed tolerance cannot tell those apart: for C(h) = exp(-3h) the
    # gap at h = 1e-8 is still 3e-8, which fails a 1e-8 test despite the
    # function being perfectly continuous.
    ratio = float(gaps[-1] / gaps[0]) if abs(gaps[0]) > 0 else 0.0
    shrinking = abs(gaps[-1]) < abs(gaps[0]) * 0.1 or abs(gaps[0]) <= tol
    plateau = abs(gaps[-1]) > tol and ratio > 0.5
    is_cont = bool(shrinking and not plateau)
    nugget = 0.0 if is_cont else float(gaps[-1])
    return RichResult(
        title="Mean-square continuity",
        summary_lines=[("C(0)", c0), ("gap at h=1e-6", float(gaps[-1])),
                       ("gap ratio 1e-6/1e-2", ratio),
                       ("continuous", is_cont)],
        payload={"is_continuous": is_cont, "c0": c0,
                 "limit_at_zero_plus": float(approach[-1]),
                 "gap": float(gaps[-1]), "nugget": nugget,
                 "gap_ratio": ratio, "gamma_limit": float(2.0 * nugget),
                 "approach": approach, "gaps": gaps, "lags": hs},
    )


def cheatsheet():
    return "spcont: MS continuous iff C is continuous at 0; a nugget breaks it."
