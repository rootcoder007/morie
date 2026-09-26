# morie.fn -- function file (rootcoder007/morie)
"""Pair correlation function g(r)"""


def pair_corr_fn(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Pair correlation function g(r)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptpcf.pair_corr_fn is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pair = pair_corr_fn


def cheatsheet() -> str:
    return "pair_corr_fn({}) -> Pair correlation function g(r)"


# compact alias per ledger/NAMING.md
paircorrfn = pair_corr_fn
