"""Binomial point process: n points independently uniform on a region."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import as_region, region_area

__all__ = ["schabenberger_binomial_process"]


def schabenberger_binomial_process(n=100, region=None, seed=None):
    r"""
    Binomial point process: a FIXED number of independent uniform points.

    The count is not random -- that is the whole difference from the
    Poisson process. For any sub-region B,

    .. math::

        N(B) \sim \mathrm{Binomial}\!\left(n,\; p =
        \frac{\nu(B)}{\nu(A)}\right)

    so :math:`E[N(B)] = np` and :math:`\mathrm{Var}[N(B)] = np(1-p)`. The
    variance is SMALLER than the mean, whereas a Poisson process has them
    equal; conditioning on the total is what removes that extra
    variability.

    Conditioning an HPP on :math:`N(A) = n` gives exactly this process
    (Sec. 3.2.3, process equivalence).

    Parameters
    ----------
    n : int
        Number of events, fixed. Must be >= 0.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices. Unit square by default.
    seed : int, optional
        Seed for the realisation.

    Returns
    -------
    RichResult
        ``points``, ``n``, ``area``, ``intensity`` (``n / area``), and for
        a stated sub-region fraction ``p``: ``binomial_mean``,
        ``binomial_var``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.2.1.
    """
    n = int(n)
    if n < 0:
        raise ValueError("`n` must be >= 0")
    reg = as_region(region, None) if region is not None else (0.0, 0.0, 1.0, 1.0)
    area = region_area(reg)
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(reg[0], reg[2], n),
                           rng.uniform(reg[1], reg[3], n)])

    def counts_in_fraction(p):
        """Mean and variance of N(B) when nu(B)/nu(A) = p."""
        return n * p, n * p * (1.0 - p)

    m, v = counts_in_fraction(0.5)
    return RichResult(
        title="Binomial point process",
        summary_lines=[("n (fixed)", n), ("area", area),
                       ("intensity", n / area if area else float("nan"))],
        payload={"points": pts, "n": n, "area": area,
                 "intensity": n / area if area else float("nan"),
                 "binomial_mean_half": m, "binomial_var_half": v,
                 "counts_in_fraction": counts_in_fraction, "region": reg},
    )


def cheatsheet():
    return "spbino: n fixed uniform points; Var[N(B)] = np(1-p) < mean."
