"""Strip probability P = rho_Z(z) dz for the sum variable.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.66).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sumdensp"]


def sumdensp(grid_x, density_x, grid_y, density_y, z, dz):
    """Strip probability P = rho_Z(z) dz for the sum variable.

    Parameters
    ----------
    grid_x, density_x, grid_y, density_y : array-like
        The two densities.
    z : float
        Strip centre.
    dz : float
        Strip width.

    Returns
    -------
    RichResult
        Keys: probability, rho_z.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.66).
    """
    rho = _morin.sum_density_convolution(grid_x, density_x, grid_y, density_y, z)
    value = rho * float(dz)
    payload = {"probability": value, "rho_z": rho}
    lines = [("P(strip)", value)]
    return RichResult(
        title="Strip probability P = rho_Z(z) dz for the sum variable.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sumdensp: Strip probability rho_Z(z) dz. Morin (2016) eq (6.66)."
