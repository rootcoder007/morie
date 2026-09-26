"""CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.."""

__all__ = ["wilcox_chapter_12_equation_9"]


def wilcox_chapter_12_equation_9(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.

    Formula: (L − 1)f1−α M SWG

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
        See ``morie.fn.describe('wilcox12e9')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.12 eq.12.9
    """
    raise NotImplementedError(
        "morie.fn.wilcox12e9.wilcox_chapter_12_equation_9 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "wilcox12e9: CentralTendency equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences."
