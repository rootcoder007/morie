"""Lord's chi-square test for differential item functioning."""

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import hypothesis_test_result

__all__ = ["lord_chi_square"]


def lord_chi_square(b_R, b_F, V_R, V_F=None):
    r"""Lord's chi-square DIF statistic for one item.

    Compares the item's parameter vector estimated separately in a
    reference and a focal group:

    .. math::

       \chi^2=(\hat\nu_R-\hat\nu_F)^{\!\top}
              \bigl(\hat\Sigma_R+\hat\Sigma_F\bigr)^{-1}
              (\hat\nu_R-\hat\nu_F)
       \;\sim\;\chi^2_{p}

    with ``p`` the number of parameters compared -- 1 for the 1PL
    (difficulty only), 2 for the 2PL, 3 for the 3PL. The covariances
    **add** because the two groups are independent samples.

    This presumes the two sets of estimates are already on a common
    metric. IRT parameters are identified only up to a linear
    transformation of theta, so if the groups were calibrated
    separately without linking, a significant result may be reporting
    the scale difference rather than DIF. Nothing in the arithmetic can
    detect that; it has to be handled before calling.

    Parameters
    ----------
    b_R, b_F : array-like
        Item parameter estimates in the reference and focal groups.
    V_R : array-like
        Covariance of ``b_R``. If ``V_F`` is omitted this is taken to be
        the already-summed covariance of the difference.
    V_F : array-like, optional
        Covariance of ``b_F``.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``pvalue``, ``df``, ``difference``.

    References
    ----------
    Lord, F. M. (1980). *Applications of Item Response Theory to
    Practical Testing Problems*. Erlbaum, ch. 14.
    Definition cross-checked against the reference implementation in
    the difR package (``LordChi2``), which forms ``solve(Sig1 + Sig2)``.
    """
    vr = np.atleast_1d(np.asarray(b_R, dtype=float)).ravel()
    vf = np.atleast_1d(np.asarray(b_F, dtype=float)).ravel()
    if vr.size != vf.size:
        raise ValueError("b_R and b_F must have the same length.")
    p = int(vr.size)
    d = vr - vf
    S = np.atleast_2d(np.asarray(V_R, dtype=float))
    if S.shape == (1, 1) and p == 1:
        pass
    if V_F is not None:
        S2 = np.atleast_2d(np.asarray(V_F, dtype=float))
        if S2.shape != S.shape:
            raise ValueError("V_R and V_F must have the same shape.")
        S = S + S2
    if S.shape != (p, p):
        raise ValueError("covariance must be %d x %d for %d parameters." % (p, p, p))
    x = np.linalg.solve(S, d)
    stat = float(sum(float(d[i]) * float(x[i]) for i in range(p)))
    if stat < 0:
        raise ValueError(
            "negative quadratic form: the summed covariance is not positive "
            "definite, so no chi-square statistic exists."
        )
    return hypothesis_test_result(
        test_name="Lord chi-square DIF",
        statistic=stat,
        pvalue=float(stats.chi2.sf(stat, p)),
        extra_summary=[("df", p)],
        extra_payload={
            "df": p,
            "difference": [float(t) for t in d],
            "method": "Lord (1980) chi-square test of item parameter equality",
        },
    )


def cheatsheet():
    return "lordzs: Lord's chi-square DIF test on item parameter differences"
