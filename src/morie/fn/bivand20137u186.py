"""Correlation expression (auto-extracted; see ref).."""

__all__ = ["bivand2013_chapter_7_unnumbered_186"]


def bivand2013_chapter_7_unnumbered_186(x, y=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation expression (auto-extracted; see ref).

    Formula: B = λW,wher eλis a spatial autocorrelation parameter andW is a matrix

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
        See ``morie.fn.describe('bivand20137u186')`` for the full guide.

    References
    ----------
    bivand2013, ch.7 (unnumbered)
    """
    raise NotImplementedError(
        "morie.fn.bivand20137u186.bivand2013_chapter_7_unnumbered_186 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bivand20137u186: Correlation expression (auto-extracted; see ref)."
