"""GeneralStatistics equation extracted from Information theory MacKay.."""

__all__ = ["information_theory_mackay_chapter_6_equation_11"]


def information_theory_mackay_chapter_6_equation_11(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Information theory MacKay.

    Formula: PL(ajx1;:::;xn1) = Fa + 1

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
        See ``morie.fn.describe('information_theory_mackay6e11')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.6 eq.6.11
    """
    raise NotImplementedError(
        "morie.fn.information_theory_mackay6e11.information_theory_mackay_chapter_6_equation_11 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "information_theory_mackay6e11: GeneralStatistics equation extracted from Information theory MacKay."
