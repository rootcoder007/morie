"""Correlation equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.."""

__all__ = ["wilcox_chapter_8_equation_4"]


def wilcox_chapter_8_equation_4(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.

    Formula: testing Equation (8.4) (using the R function yuend described momentarily). With ρ = 0

    Parameters
    ----------
    x : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``morie.fn.describe('wilcox8e4')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.8 eq.8.4
    """
    raise NotImplementedError(
        "morie.fn.wilcox8e4.wilcox_chapter_8_equation_4 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "wilcox8e4: Correlation equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences."
