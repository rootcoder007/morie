"""Association expression (auto-extracted; see ref).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_934"]


def chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_934(x, y=None):
    """
    Association expression (auto-extracted; see ref).

    Formula: forms. Moreover, methods show similar performance in terms of Kendall’s τ correlation when γ = 1

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
        See ``morie.fn.describe('chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u934')`` for the full guide.

    References
    ----------
    [Chapman & Hall CRC Handbooks of Modern Statistical Methods] José R. Zubizarreta, Elizabeth A. Stuart, Dylan S. Small, Paul R - Handbook of Matching and Weighting Adjustments for Causal Inference, ch.10 (unnumbered)
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
    return "chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u934: Association expression (auto-extracted; see ref)."
