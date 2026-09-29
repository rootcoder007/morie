"""Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.

Implements eq (6.64) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["joint_density_factorizes"]


def joint_density_factorizes(grid_x, density_x, grid_y, density_y):
    """Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.64).

    Examples
    --------
    >>> round(joint_density_factorizes([0, 1, 2], [0.5, 0.5, 0.5], [0, 1], [1.0, 1.0])["total_mass"], 12)
    1.0
    """
    joint, total = _morin.joint_density_factorizes(grid_x, density_x, grid_y, density_y)
    payload = {"total_mass": total, "shape": list(joint.shape)}
    lines = [("total mass", total)]
    return RichResult(
        title="Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e64: Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y. Morin (2016) eq (6.64)."
