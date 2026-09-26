"""Regression equation extracted from Use R.."""

__all__ = ["use_r_chapter_8_equation_5"]


def use_r_chapter_8_equation_5(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Regression equation extracted from Use R.

    Formula: [EQ] Xj(s)βj + e(s)= Xβ + e(s), (8.5)

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
        See ``morie.fn.describe('use_r8e5')`` for the full guide.

    References
    ----------
    Use R, ch.8 eq.8.5
    """
    raise NotImplementedError(
        "morie.fn.use_r8e5.use_r_chapter_8_equation_5 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "use_r8e5: Regression equation extracted from Use R."
