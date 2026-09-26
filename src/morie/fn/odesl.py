"""Symbolic ODE solver."""

__all__ = ["ode_symbolic"]


def ode_symbolic(ode):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Symbolic ODE solver

    Formula: classify (separable, linear, exact, Bernoulli)

    Parameters
    ----------
    ode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bronstein (1997) Symbolic Integration
    """
    raise NotImplementedError(
        "morie.fn.odesl.ode_symbolic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "odesl: Symbolic ODE solver"


# compact alias per ledger/NAMING.md
odesymbolic = ode_symbolic
