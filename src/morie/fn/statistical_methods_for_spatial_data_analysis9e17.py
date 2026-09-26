"""Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_9_equation_17"]


def statistical_methods_for_spatial_data_analysis_chapter_9_equation_17(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.

    Formula: σ2           τ exp{−(k/c)(τ 2 + θ2 )p }

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis9e17')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.9 eq.9.17
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis9e17.statistical_methods_for_spatial_data_analysis_chapter_9_equation_17 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis9e17: Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis."
