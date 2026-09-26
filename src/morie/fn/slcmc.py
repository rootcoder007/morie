"""Slice sampling."""

__all__ = ["slice_sampler"]


def slice_sampler(log_p, x0, width):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Slice sampling

    Formula: alternate sampling u | x and x | u

    Parameters
    ----------
    log_p : array-like
        Input data.
    x0 : array-like
        Input data.
    width : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neal (2003)
    """
    raise NotImplementedError(
        "morie.fn.slcmc.slice_sampler is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "slcmc: Slice sampling"


# compact alias per ledger/NAMING.md
slicesampler = slice_sampler
