# morie.fn -- function file (rootcoder007/morie)
"""Prediction variance under compound symmetry, and when it beats independence."""

from ._richresult import RichResult

__all__ = [
    "cspvar",
    "statistical_methods_for_spatial_data_analysis_chapter_1_equation_27",
]


def cspvar(n, rho, sigma2=1.0):
    r"""Compound-symmetry prediction variance and the precision condition.

    For :math:`\sigma^{-2}\boldsymbol\Sigma = (1-\rho)\mathbf{I} +
    \rho\mathbf{J}` the prediction error variance of a new observation
    is (p. 34)

    .. math::

        \sigma^2_{pred} = \sigma^2\left[1 + \frac{1}{n}
        \frac{(\rho n)^2 + (1-\rho)^2}{1 + (n-1)\rho}\right],

    against :math:`\sigma^2(1 + 1/n)` when the data are uncorrelated.
    The correlated model is the more precise one exactly when the middle
    term is below one, which for :math:`\rho > 0` is (eq. 1.27)

    .. math::  \rho < \frac{n+1}{n^2+1}.

    Equicorrelation also bounds ``rho`` from below: ``Var[sum Y_i] =
    n sigma^2 (1 + (n-1) rho) > 0`` forces ``rho > -1/(n-1)``.

    Parameters
    ----------
    n : int
        Sample size, at least 2.
    rho : float
        Equicorrelation, in ``(-1/(n-1), 1]``.
    sigma2 : float
        Marginal variance ``sigma^2``, positive.

    Returns
    -------
    RichResult
        ``var_pred``, ``var_indep``, ``ratio_term``, ``threshold``,
        ``more_precise``, ``rho_lower_bound``, ``n``, ``rho``, ``sigma2``.
        ``more_precise`` is the decision "the compound-symmetry model
        predicts more precisely than the independence model".

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, Sec. 1.5.1, eq. (1.27),
    p. 34.
    """
    n = int(n)
    rho = float(rho)
    sigma2 = float(sigma2)
    if n < 2:
        raise ValueError("`n` must be at least 2")
    if sigma2 <= 0:
        raise ValueError("`sigma2` must be positive")
    lower = -1.0 / (n - 1)
    if rho <= lower or rho > 1:
        raise ValueError("`rho` must lie in (-1/(n-1), 1]")

    term = ((rho * n) ** 2 + (1.0 - rho) ** 2) / (1.0 + (n - 1) * rho)
    var_pred = sigma2 * (1.0 + term / n)
    var_indep = sigma2 * (1.0 + 1.0 / n)
    threshold = (n + 1.0) / (n * n + 1.0)

    return RichResult(
        title="Compound-symmetry prediction variance (eq. 1.27)",
        summary_lines=[("n", n), ("rho", rho), ("var_pred", var_pred)],
        payload={
            "var_pred": var_pred,
            "var_indep": var_indep,
            "ratio_term": term,
            "threshold": threshold,
            "more_precise": bool(var_pred < var_indep),
            "rho_lower_bound": lower,
            "n": n,
            "rho": rho,
            "sigma2": sigma2,
        },
    )


statistical_methods_for_spatial_data_analysis_chapter_1_equation_27 = cspvar


def cheatsheet():
    return "cspvar: sigma2_pred under compound symmetry; more precise iff rho < (n+1)/(n^2+1)."
