"""Correlation expression (auto-extracted; see ref).."""

__all__ = ["statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_99"]


def statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_99(x, y=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Correlation expression (auto-extracted; see ref).

    Formula: equicorrelation. Applying Theorem 1.1 leads to 1 Σ−1 1 = nσ −2 /[1+(n−1)ρ].

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``morie.fn.describe('statistical_methods_for_spatial_data_analysis1u99')`` for the full guide.

    References
    ----------
    Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.1 (unnumbered)
    """
    raise NotImplementedError(
        "morie.fn.statistical_methods_for_spatial_data_analysis1u99.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_99 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "statistical_methods_for_spatial_data_analysis1u99: Correlation expression (auto-extracted; see ref)."
