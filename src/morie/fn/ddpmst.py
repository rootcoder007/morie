"""DDPM reverse step."""

__all__ = ["ddpm_step"]


def ddpm_step(x_t, t, eps_theta):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    DDPM reverse step

    Formula: x_{t-1} = (1/sqrt(alpha_t)) (x_t - eps_theta) + sigma_t z

    Parameters
    ----------
    x_t : array-like
        Input data.
    t : array-like
        Input data.
    eps_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ho-Jain-Abbeel (2020)
    """
    raise NotImplementedError(
        "morie.fn.ddpmst.ddpm_step is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ddpmst: DDPM reverse step"


# compact alias per ledger/NAMING.md
ddpmstep = ddpm_step
