"""Association expression (auto-extracted; see ref).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_245"]


def analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_245(x, y=None):
    """
    Association expression (auto-extracted; see ref).

    Formula: thin=50, burnin=5000, normalize=FALSE,

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
        See ``morie.fn.describe('analyzing_spatial_models_of_choice_and_judgment5u245')`` for the full guide.

    References
    ----------
    Cahoon, Hinich & Ordeshook (1978) Analyzing Spatial Models of Choice and Judgment, ch.5 (unnumbered)
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
    return "analyzing_spatial_models_of_choice_and_judgment5u245: Association expression (auto-extracted; see ref)."
