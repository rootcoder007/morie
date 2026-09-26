"""GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R.."""

__all__ = ["hedderich_chapter_2_equation_49"]


def hedderich_chapter_2_equation_49(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R.

    Formula: 2. The coefficient matrix A has full column rankrg(An,m) = m<n

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
        See ``morie.fn.describe('hedderich2e49')`` for the full guide.

    References
    ----------
    Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R, ch.2 eq.2.49
    """
    raise NotImplementedError(
        "morie.fn.hedderich2e49.hedderich_chapter_2_equation_49 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hedderich2e49: GeneralStatistics equation extracted from Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R."
