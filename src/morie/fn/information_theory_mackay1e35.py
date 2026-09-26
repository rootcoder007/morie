"""GeneralStatistics equation extracted from Information theory MacKay.."""

__all__ = ["information_theory_mackay_chapter_1_equation_35"]


def information_theory_mackay_chapter_1_equation_35(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Information theory MacKay.

    Formula: [EQ] C(f) = 1H2(f) = 1

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
        See ``morie.fn.describe('information_theory_mackay1e35')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.1 eq.1.35
    """
    raise NotImplementedError(
        "morie.fn.information_theory_mackay1e35.information_theory_mackay_chapter_1_equation_35 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "information_theory_mackay1e35: GeneralStatistics equation extracted from Information theory MacKay."
