"""Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17"]


def andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17(x, y=None):
    """
    Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.

    Formula: distribution for φ is U(φmin, φmax). As noted by Stern and Cressie (1999) this

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``morie.fn.describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17')`` for the full guide.

    References
    ----------
    Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.6 eq.6.17
    """
    if y is None:
        # Auto-extracted single-input stub: correlate x against itself so
        # the call is well-defined instead of raising UnboundLocalError.
        y = x
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={
                "n": n,
                "method": "Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
            },
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={
            "n": n,
            "method": "Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
            "p_value": float(result.pvalue),
        },
    )


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17: Correlation equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling."
