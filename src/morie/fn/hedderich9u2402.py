"""Correlation expression (auto-extracted; see ref).."""

__all__ = ["hedderich_chapter_9_unnumbered_2402"]


def hedderich_chapter_9_unnumbered_2402(x, y=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation expression (auto-extracted; see ref).

    Formula: The individual hypothesesHi0 :µi =µ0 are to be rejected as soon as the value of the test statistic|Di|is greater than the corresponding quantile of the multivariate t-distributiontν,k,R,1−α.

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
        See ``morie.fn.describe('hedderich9u2402')`` for the full guide.

    References
    ----------
    Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R, ch.9 (unnumbered)
    """
    raise NotImplementedError(
        "morie.fn.hedderich9u2402.hedderich_chapter_9_unnumbered_2402 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hedderich9u2402: Correlation expression (auto-extracted; see ref)."
