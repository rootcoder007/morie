"""Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_8_equation_1"]


def statistical_methods_for_spatial_data_analysis_chapter_8_equation_1(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.

    Formula: Corr[Z(si ), Z(sj )] = exp {−θ1 ||si − sj || exp {θ2 |ci − cj | + θ3 min[ci , cj ]}}

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis8e1')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.8 eq.8.1
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis8e1.statistical_methods_for_spatial_data_analysis_chapter_8_equation_1 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis8e1: Correlation equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis."
