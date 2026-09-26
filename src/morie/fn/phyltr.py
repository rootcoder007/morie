"""Neighbor-joining tree."""

__all__ = ["phylogenetic_tree_nj"]


def phylogenetic_tree_nj(distance_matrix):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Neighbor-joining tree

    Formula: iteratively join pair minimizing Q-metric

    Parameters
    ----------
    distance_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Saitou-Nei (1987)
    """
    raise NotImplementedError(
        "morie.fn.phyltr.phylogenetic_tree_nj is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "phyltr: Neighbor-joining tree"
