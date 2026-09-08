"""Association expression (auto-extracted; see ref).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_10"]


def neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_10(x, y=None):
    """
    Association expression (auto-extracted; see ref).

    Formula: 5.015 1.815 * 0.203*tt tAutoTheft Time Time=+ +

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
        See ``morie.fn.describe('neighborhood_structures_and_crime_a_spatial_analysis_george_2u10')`` for the full guide.

    References
    ----------
    Neighborhood Structures And Crime A Spatial Analysis George Kikuchi, ch.2 (unnumbered)
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
            test_name="Association expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Association expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Association expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={
            "n": n,
            "method": "Association expression (auto-extracted; see ref).",
            "p_value": float(result.pvalue),
        },
    )


def cheatsheet():
    return "neighborhood_structures_and_crime_a_spatial_analysis_george_2u10: Association expression (auto-extracted; see ref)."
