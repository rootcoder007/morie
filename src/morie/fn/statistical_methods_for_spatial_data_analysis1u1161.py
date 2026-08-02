"""Correlation expression (auto-extracted; see ref).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1161"]


def statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1161(x, y=None):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: F (z1 , z2 ) = (2π)−1 c[c2 + z12 + z22 ]−3/2 , −∞ < z1 , z2 , < ∞, c > 0.

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis1u1161')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.1 (unnumbered)
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
            test_name="Correlation expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={
            "n": n,
            "method": "Correlation expression (auto-extracted; see ref).",
            "p_value": float(result.pvalue),
        },
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis1u1161: Correlation expression (auto-extracted; see ref)."
