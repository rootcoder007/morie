"""Correlation expression (auto-extracted; see ref).."""

__all__ = ["use_r_chapter_2_unnumbered_121"]


def use_r_chapter_2_unnumbered_121(x, y=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation expression (auto-extracted; see ref).

    Formula: ˆγ(hi) − γ(hi)a n dˆγ(hj) − γ(hj) will be correlated, because ˆγ(hi)a n d ˆγ(hj)

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
        See ``morie.fn.describe('use_r2u121')`` for the full guide.

    References
    ----------
    Use R, ch.2 (unnumbered)
    """
    raise NotImplementedError(
        "morie.fn.use_r2u121.use_r_chapter_2_unnumbered_121 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "use_r2u121: Correlation expression (auto-extracted; see ref)."
