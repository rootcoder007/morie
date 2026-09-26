"""
GWR bandwidth selection

Category: SpatReg2
"""


def srgwb(X=None, y=None, w=None, n=50, k=3):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR bandwidth selection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.srgwb.srgwb is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "srgwb"
alias = "srgwb"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
srgwb = srgwb


def cheatsheet() -> str:
    return "srgwb({}) -> GWR bandwidth selection"
