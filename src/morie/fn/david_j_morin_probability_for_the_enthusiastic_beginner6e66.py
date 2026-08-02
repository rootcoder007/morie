"""Strip probability P = rho_z(z) dz for the sum variable.

Implements eq (6.66) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66(grid_x, density_x, grid_y, density_y, z, dz):
    """Strip probability P = rho_z(z) dz for the sum variable.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.66).
    """
    rho = _morin.sum_density_convolution(grid_x, density_x,
                                         grid_y, density_y, z)
    value = rho * float(dz)
    payload = {"probability": value, "rho_z": rho}
    lines = [("P(strip)", value)]
    return RichResult(
        title="Strip probability P = rho_z(z) dz for the sum variable.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e66: Strip probability P = rho_z(z) dz for the sum variable. Morin (2016) eq (6.66)."
