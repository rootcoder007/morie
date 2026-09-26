"""Shunting-yard infix->RPN."""

__all__ = ["shunting_yard"]


def shunting_yard(tokens):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Shunting-yard infix->RPN

    Formula: two-stack precedence-based parser

    Parameters
    ----------
    tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dijkstra (1961)
    """
    raise NotImplementedError(
        "morie.fn.shYa.shunting_yard is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "shYa: Shunting-yard infix->RPN"


# compact alias per ledger/NAMING.md
shuntingyard = shunting_yard
