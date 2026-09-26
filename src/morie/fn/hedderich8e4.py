"""GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R.."""

__all__ = ["hedderich_chapter_8_equation_4"]


def hedderich_chapter_8_equation_4(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R.

    Formula: [EQ] Yi =β0 +β1xi +ϵi i = 1,...,n (8.4)

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
        See ``morie.fn.describe('hedderich8e4')`` for the full guide.

    References
    ----------
    Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R, ch.8 eq.8.4
    """
    raise NotImplementedError(
        "morie.fn.hedderich8e4.hedderich_chapter_8_equation_4 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hedderich8e4: GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R."
