# morie.fn -- function file (rootcoder007/morie)
"""Standardised z-score of Moran's I."""

import math

from ._containers import SpatialResult


def _pnorm_upper(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mizval(I, E_I, Var_I, alternative="greater"):  # noqa: E741
    r"""Standardised Moran's I, ``z = (I - E[I]) / sqrt(Var[I])``, with its normal p-value.

    ``alternative`` is ``"greater"`` (the ``spdep::moran.test`` default),
    ``"less"`` or ``"two.sided"``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> r = mizval(0.4, -0.05, 0.02)
    >>> round(r.statistic, 12)
    3.181980515339
    """
    i_val, E_I, Var_I = float(I), float(E_I), float(Var_I)
    if not Var_I > 0:
        raise ValueError("Var_I must be positive")
    z = (i_val - E_I) / math.sqrt(Var_I)
    if alternative == "greater":
        p = _pnorm_upper(z)
    elif alternative == "less":
        p = _pnorm_upper(-z)
    elif alternative == "two.sided":
        p = 2.0 * _pnorm_upper(abs(z))
    else:
        raise ValueError("alternative must be 'greater', 'less' or 'two.sided'")
    return SpatialResult(
        name="mizval",
        statistic=z,
        p_value=p,
        expected=E_I,
        variance=Var_I,
        extra={"I": i_val, "alternative": alternative},
    )


mizval_fn = mizval


def cheatsheet() -> str:
    return "mizval(I, E_I, Var_I, alternative='greater') -> z = (I - E[I])/sqrt(Var[I]) and its normal p-value."
