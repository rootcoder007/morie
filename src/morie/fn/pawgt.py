# morie.fn -- function file (rootcoder007/morie)
"""Weighted aggregation spatial.

Category: Spatial
"""


def pawgt(data=None, positions=None, weights=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Weighted aggregation spatial.

    Parameters
    ----------
    data : array_like, optional
        Data values.
    positions : array_like, optional
        Position values (unused, kept for API compat).
    weights : array_like, optional
        Weights for aggregation.
    n : int
        Number of random samples if data is None.

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.pawgt.pawgt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "pawgt"
alias = "pawgt"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
pawgt = pawgt


def cheatsheet() -> str:
    return "pawgt({}) -> Weighted aggregation spatial."
