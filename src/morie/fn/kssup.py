"""Kolmogorov-Smirnov one-sample supremum test."""

from __future__ import annotations

from . import _array_core as np
from . import _frame_core as pd
from . import _stats_core as stats

from ._containers import TestResult
from ._helpers import _extract_col


def ks_supremum(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    dist: str = "norm",
    alternative: str = "two-sided",
) -> TestResult:
    """Kolmogorov-Smirnov one-sample supremum test.

    Tests whether the data come from a specified continuous distribution by
    computing the supremum of the absolute difference between the empirical
    and theoretical CDFs.

    Parameters
    ----------
    data : DataFrame or array
        Input data.
    col : str
        Column name if *data* is a DataFrame.
    dist : str
        Scipy distribution name (e.g. ``'norm'``, ``'expon'``, ``'uniform'``).
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    TestResult

    Notes
    -----
    Parameters of ``dist`` are FITTED from the data before the test.
    The p-value from the classical KS distribution is then
    conservative (the fitted CDF hugs the sample, shrinking D); for
    the Gaussian case with a correct null distribution use
    :func:`morie.fn.lilf.lilliefors_test`.

    References
    ----------
    Kolmogorov, A. N. (1933). Sulla determinazione empirica di una
    legge di distribuzione. *Giorn. Ist. Ital. Attuari*, 4, 83-91.
    Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of
    fit. *JASA*, 46(253), 68-78.
    """
    x = _extract_col(data, col)
    if len(x) < 5:
        raise ValueError("Need at least 5 observations for KS test")
    try:
        dist_obj = getattr(stats, dist)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {dist}")
    params = dist_obj.fit(x)
    # Freeze the distribution rather than passing the name + args: recent
    # scipy maps some named CDFs to bare special functions (norm -> ndtr)
    # that reject location/scale arguments.
    result = stats.kstest(x, dist_obj(*params).cdf, alternative=alternative)
    return TestResult(
        test_name=f"KS supremum test ({dist})",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method=f"Kolmogorov-Smirnov vs {dist}",
        n=len(x),
        extra={"dist": dist, "fit_params": params, "alternative": alternative},
    )


kssup = ks_supremum


def cheatsheet() -> str:
    return "ks_supremum({}) -> Supremum test / KS statistic."
