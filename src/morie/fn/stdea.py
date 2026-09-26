"""Differential equation spatio-temporal covariance"""


def st_diff_equation(coords=None, times=None, data=None, *, n=30, diffusion=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatio-temporal covariance via diffusion differential equation.

    Models spatio-temporal dependence through a diffusion PDE:
    dZ/dt = D * nabla^2(Z) + noise

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.stdea.st_diff_equation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "stdea"
alias = "st_diff_equation"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
st_diff_equation = st_diff_equation


def cheatsheet() -> str:
    return "st_diff_equation({}) -> Differential equation spatio-temporal covariance"


# compact alias per ledger/NAMING.md
stdiffequation = st_diff_equation
