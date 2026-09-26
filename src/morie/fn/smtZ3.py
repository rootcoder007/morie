"""SMT solver framework."""

__all__ = ["smt_solver"]


def smt_solver(formula):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SMT solver framework

    Formula: DPLL(T) -- SAT + theory solvers

    Parameters
    ----------
    formula : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Moura-Bjørner (2008) Z3
    """
    raise NotImplementedError(
        "morie.fn.smtZ3.smt_solver is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "smtZ3: SMT solver framework"


# compact alias per ledger/NAMING.md
smtsolver = smt_solver
