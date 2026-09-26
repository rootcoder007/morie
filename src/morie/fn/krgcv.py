# morie.fn -- function file (rootcoder007/morie)
"""
Kriging cross-validation

Category: KrigFilt
"""


def krgcv(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging cross-validation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.krgcv.krgcv is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "krgcv"
alias = "krgcv"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
krgcv = krgcv


def cheatsheet() -> str:
    return "krgcv({}) -> Kriging cross-validation"
