"""Correlation expression (auto-extracted; see ref).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2504"]


def probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2504(x, y=None):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: definite functions ρ with ρ( 0) = 1 are autocorrelation functions of stationary sequences

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
        See ``morie.fn.describe('probability_and_random_processes_with_one_thousand_exercises5u2504')`` for the full guide.

    References
    ----------
    Probability and Random Processes with One Thousand Exercises -- Geoffrey  Stirzaker Grimmett, ch.5 (unnumbered)
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
    return "probability_and_random_processes_with_one_thousand_exercises5u2504: Correlation expression (auto-extracted; see ref)."
