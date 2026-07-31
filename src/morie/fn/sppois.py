"""Poisson process: independent counts, N(A) ~ Pois(lambda |A|)."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import as_region, region_area

__all__ = ["schabenberger_poisson_process"]


def schabenberger_poisson_process(lam=1.0, region=None, seed=None):
    r"""
    Homogeneous Poisson process (HPP), the CSR reference model.

    Two defining properties (Sec. 3.2.2):

    * :math:`N(A) \sim \mathrm{Poisson}(\lambda\,\nu(A))` for any region A;
    * counts in DISJOINT regions are independent.

    Conditional on :math:`N(A) = n`, the n events are independently and
    uniformly distributed over A -- which is exactly the binomial process
    of Sec. 3.2.1, and is how an HPP is simulated.

    Parameters
    ----------
    lam : float
        Intensity :math:`\lambda`, events per unit area. Must be > 0.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices. Unit square by default.
    seed : int, optional
        Seed for the realisation.

    Returns
    -------
    RichResult
        ``points`` (the realisation), ``n`` (its count), ``lambda``,
        ``area``, ``expected_n`` (:math:`\lambda\nu(A)`), ``var_n``
        (equal to the mean, as the Poisson requires).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.2.2.
    """
    if lam <= 0:
        raise ValueError("`lam` must be > 0")
    reg = as_region(region, None) if region is not None else (0.0, 0.0, 1.0, 1.0)
    area = region_area(reg)
    rng = np.random.default_rng(seed)
    n = int(rng.poisson(lam * area))
    pts = np.column_stack([rng.uniform(reg[0], reg[2], n),
                           rng.uniform(reg[1], reg[3], n)])
    return RichResult(
        title="Homogeneous Poisson process",
        summary_lines=[("lambda", float(lam)), ("area", area), ("n", n)],
        payload={"points": pts, "n": n, "lambda": float(lam), "area": area,
                 "expected_n": float(lam) * area, "var_n": float(lam) * area,
                 "region": reg},
    )


def cheatsheet():
    return "sppois: HPP; N(A)~Pois(lambda|A|), independent on disjoint sets."
