"""GeneralStatistics equation extracted from Design of observational studies.."""

__all__ = ["design_of_observational_studies_chapter_3_equation_1"]


def design_of_observational_studies_chapter_3_equation_1(x, cdf=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GeneralStatistics equation extracted from Design of observational studies.

    Formula: Pr ( Z1 = z1,...,Z L = zL| rT 1,r C1, x1,Z 1,...,r TL ,r CL , xL,Z L)

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
        See ``morie.fn.describe('design_of_observational_studies3e1')`` for the full guide.

    References
    ----------
    Design of observational studies, ch.3 eq.3.1
    """
    raise NotImplementedError(
        "morie.fn.design_of_observational_studies3e1.design_of_observational_studies_chapter_3_equation_1 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return (
        "design_of_observational_studies3e1: GeneralStatistics equation extracted from Design of observational studies."
    )
