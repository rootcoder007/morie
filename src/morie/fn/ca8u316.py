"""Correlation expression involving 'power' (auto-extracted; see reference for full context).."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import hypothesis_test_result

__all__ = ["ca_chapter_8_unnumbered_316"]


def ca_chapter_8_unnumbered_316(x, y=None):
    """
    Correlation expression involving 'power' (auto-extracted; see reference for full context).

    Formula: pwr.r.test(n = NULL, r = 0.3, sig.level = 0.05,

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
        See ``morie.fn.describe('ca8u316')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.8 (unnumbered)
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
            test_name="Correlation expression involving 'power' (auto-extracted; see reference for full context).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={
                "n": n,
                "method": "Correlation expression involving 'power' (auto-extracted; see reference for full context).",
            },
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression involving 'power' (auto-extracted; see reference for full context).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={
            "n": n,
            "method": "Correlation expression involving 'power' (auto-extracted; see reference for full context).",
            "p_value": float(result.pvalue),
        },
    )


def cheatsheet():
    return "ca8u316: Correlation expression involving 'power' (auto-extracted; see reference for full context)."
