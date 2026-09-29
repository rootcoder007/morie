# morie.fn -- function file (rootcoder007/morie)
"""Variance of Moran's I under the null (normality or randomisation assumption)."""

from ._containers import SpatialResult


def mivar(n, S0, S1, S2, b2=None):
    r"""Variance of Moran's I under the null of no spatial autocorrelation.

    With the weights constants ``S0 = sum_ij w_ij``, ``S1 = 0.5 sum_ij (w_ij +
    w_ji)^2`` and ``S2 = sum_i (w_i. + w_.i)^2`` the normality variance is

    ``Var[I] = (n^2 S1 - n S2 + 3 S0^2) / (S0^2 (n^2 - 1)) - 1 / (n - 1)^2``.

    If the sample kurtosis ``b2 = n sum z^4 / (sum z^2)^2`` is supplied the
    randomisation variance is returned instead,

    ``Var[I] = [n((n^2 - 3n + 3) S1 - n S2 + 3 S0^2) - b2((n^2 - n) S1 - 2n S2
    + 6 S0^2)] / ((n - 1)(n - 2)(n - 3) S0^2) - 1 / (n - 1)^2``

    (Cliff and Ord 1981, eqs. 1.36 and 1.38; ``spdep::moran.test``).

    Parameters
    ----------
    n : int
        Number of observations.
    S0, S1, S2 : float
        Weights constants.
    b2 : float, optional
        Sample kurtosis; ``None`` selects the normality variance.

    Returns
    -------
    SpatialResult
        ``statistic`` and ``variance`` are ``Var[I]``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> round(mivar(4, 6.0, 12.0, 40.0).statistic, 12)
    0.148148148148
    """
    n = float(n)
    S0, S1, S2 = float(S0), float(S1), float(S2)
    if b2 is None:
        v = (n * n * S1 - n * S2 + 3.0 * S0 * S0) / (S0 * S0 * (n * n - 1.0)) - 1.0 / ((n - 1.0) ** 2)
        kind = "normality"
    else:
        b2 = float(b2)
        num = n * ((n * n - 3.0 * n + 3.0) * S1 - n * S2 + 3.0 * S0 * S0) - b2 * (
            (n * n - n) * S1 - 2.0 * n * S2 + 6.0 * S0 * S0
        )
        v = num / ((n - 1.0) * (n - 2.0) * (n - 3.0) * S0 * S0) - 1.0 / ((n - 1.0) ** 2)
        kind = "randomisation"
    return SpatialResult(name="mivar", statistic=v, variance=v, extra={"assumption": kind})


mivar_fn = mivar


def cheatsheet() -> str:
    return "mivar(n, S0, S1, S2, b2=None) -> Var[I] of Moran's I (normality, or randomisation given b2)."
