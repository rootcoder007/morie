"""CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.."""

__all__ = ["wilcox_chapter_5_equation_5"]


def wilcox_chapter_5_equation_5(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.

    Formula: • ForH0: µ ≥ µ 0, reject if T ≤ c, where c is the α quantile of Student’s T distribution

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
        See ``morie.fn.describe('wilcox5e5')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.5 eq.5.5
    """
    raise NotImplementedError(
        "morie.fn.wilcox5e5.wilcox_chapter_5_equation_5 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "wilcox5e5: CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences."
