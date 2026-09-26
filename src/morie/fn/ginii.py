# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Gini impurity for classification tree splits."""

__all__ = ["gini_impurity"]


def gini_impurity(class_probs):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Gini impurity for classification tree splits

    Formula: Gini(t) = 1 - sum_k p_k^2; p_k = proportion class k at node t

    Parameters
    ----------
    class_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gini': 'float'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    raise NotImplementedError(
        "morie.fn.ginii.gini_impurity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "giniI: Gini impurity for classification tree splits"


# compact alias per ledger/NAMING.md
giniimpurity = gini_impurity
