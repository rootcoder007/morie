"""Maximum entropy distribution."""

__all__ = ["max_entropy"]


def max_entropy(constraints):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Maximum entropy distribution

    Formula: argmax H(p) s.t. moment constraints

    Parameters
    ----------
    constraints : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaynes (1957)
    """
    raise NotImplementedError(
        "morie.fn.mxtent.max_entropy is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "mxtent: Maximum entropy distribution"


# compact alias per ledger/NAMING.md
maxentropy = max_entropy
