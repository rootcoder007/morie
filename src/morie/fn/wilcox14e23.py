"""GeneralStatistics equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.."""

__all__ = ["wilcox_chapter_14_equation_23"]


def wilcox_chapter_14_equation_23(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.

    Formula: [EQ] Y =β 0 +β 1X1 +β 2X2 +β 3X1X2 +e. (14.22)

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
        See ``morie.fn.describe('wilcox14e23')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.14 eq.14.23
    """
    raise NotImplementedError(
        "morie.fn.wilcox14e23.wilcox_chapter_14_equation_23 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "wilcox14e23: GeneralStatistics equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences."
