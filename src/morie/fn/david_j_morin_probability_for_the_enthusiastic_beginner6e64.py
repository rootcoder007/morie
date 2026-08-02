"""Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.

Implements eq (6.64) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_64"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_64(grid_x, density_x, grid_y, density_y):
    """Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.64).
    """
    joint, total = _morin.joint_density_factorizes(grid_x, density_x,
                                                   grid_y, density_y)
    payload = {"total_mass": total, "shape": list(joint.shape)}
    lines = [("total mass", total)]
    return RichResult(
        title="Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e64: Joint density of independent variables factorizes: rho(x,y) = rho_x rho_y. Morin (2016) eq (6.64)."
