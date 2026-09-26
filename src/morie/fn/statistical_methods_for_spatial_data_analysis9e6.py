"""Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_9_equation_6"]


def statistical_methods_for_spatial_data_analysis_chapter_9_equation_6(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis.

    Formula: Cov[Z(s, t), Z(s + h, t + k) = σ 2 (t) Cs (h).            (9.6)

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
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis9e6')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.9 eq.9.6
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis9e6.statistical_methods_for_spatial_data_analysis_chapter_9_equation_6 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis9e6: Dispersion equation extracted from Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis."
