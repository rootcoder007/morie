"""GeneralStatistics equation extracted from Information theory MacKay.."""

__all__ = ["information_theory_mackay_chapter_8_equation_6"]


def information_theory_mackay_chapter_8_equation_6(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Information theory MacKay.

    Formula: [EQ] h(x;y) =h(x) +h(yjx): (8.6)

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
        See ``morie.fn.describe('information_theory_mackay8e6')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.8 eq.8.6
    """
    raise NotImplementedError(
        "morie.fn.information_theory_mackay8e6.information_theory_mackay_chapter_8_equation_6 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "information_theory_mackay8e6: GeneralStatistics equation extracted from Information theory MacKay."
