"""Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_9_equation_2"]


def statistical_methods_for_spatial_data_analysis_chapter_9_equation_2(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.

    Formula: Corr[Z(si , ti ), Z(sj , tj )] = R(θs ||hij ||2 + θt k 2 ).        (9.3)

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis9e2')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.9 eq.9.2
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis9e2.statistical_methods_for_spatial_data_analysis_chapter_9_equation_2 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis9e2: Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis."
