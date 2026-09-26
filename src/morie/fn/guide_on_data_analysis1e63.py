"""GeneralStatistics equation extracted from guide on data analysis.."""

__all__ = ["guide_on_data_analysis_chapter_1_equation_63"]


def guide_on_data_analysis_chapter_1_equation_63(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from guide on data analysis.

    Formula: export_summs(fit, fit_b, fit_c, robust = "HC3", coefs = coef_names)

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
        See ``morie.fn.describe('guide_on_data_analysis1e63')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.1 eq.1.63
    """
    raise NotImplementedError(
        "morie.fn.guide_on_data_analysis1e63.guide_on_data_analysis_chapter_1_equation_63 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "guide_on_data_analysis1e63: GeneralStatistics equation extracted from guide on data analysis."
