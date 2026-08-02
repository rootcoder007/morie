"""Correlation expression (auto-extracted; see ref).."""

from . import _array_core as np
from scipy import stats

from ._richresult import hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602"]


def bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602(x, y=None):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: 0. 95. The response Y was generated according to Pr(Y = 1|x1≤ 0. 5) = 0. 2,

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
        See ``morie.fn.describe('bookadvanced_elementsofstatisticallearning3u602')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.3 (unnumbered)
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
    return "bookadvanced_elementsofstatisticallearning3u602: Correlation expression (auto-extracted; see ref)."
