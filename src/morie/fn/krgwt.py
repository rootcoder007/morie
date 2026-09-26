# morie.fn -- function file (rootcoder007/morie)
"""
Kriging weight map

Category: KrigFilt
"""


def krgwt(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging weight map

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.krgwt.krgwt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "krgwt"
alias = "krgwt"
quote = "It is not what happens to you, but how you react, that matters. -- Epictetus"
krgwt = krgwt


def cheatsheet() -> str:
    return "krgwt({}) -> Kriging weight map"
