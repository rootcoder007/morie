"""Spatial equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_6_equation_9"]


def statistical_methods_for_spatial_data_analysis_chapter_6_equation_9(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Spatial equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.

    Formula: Z(s) = X(s)β(s0 ) + e0 ,       e0 ∼ (0, σ 2 W(s0 )−1 ),            (6.9)

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis6e9')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.6 eq.6.9
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis6e9.statistical_methods_for_spatial_data_analysis_chapter_6_equation_9 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis6e9: Spatial equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis."
