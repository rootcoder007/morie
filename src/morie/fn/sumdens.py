"""Density of the sum of two independent variables, by convolution.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.65).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sumdens"]


def sumdens(grid_x, density_x, grid_y, density_y, z):
    """Density of the sum of two independent variables, by convolution.

    rho_Z(z) = integral rho_X(x) rho_Y(z - x) dx, trapezoid on the x
    grid with rho_Y linearly interpolated and zero outside its grid.

    Parameters
    ----------
    grid_x, density_x : array-like
        The density of X.
    grid_y, density_y : array-like
        The density of Y.
    z : float
        Evaluation point.

    Returns
    -------
    RichResult
        Keys: z, density.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.65).
    """
    value = _morin.sum_density_convolution(grid_x, density_x, grid_y, density_y, z)
    payload = {"z": float(z), "density": value}
    lines = [("rho_Z(z)", value)]
    return RichResult(
        title="Density of the sum of two independent variables, by convolution.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sumdens: Convolution density of X + Y. Morin (2016) eq (6.65)."
