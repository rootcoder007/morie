# morie.fn -- function file (rootcoder007/morie)
"""Gravity model calibration (beta estimation)."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igravcl(flows, mass_o, mass_d, dist):
    r"""Gravity model calibration (beta estimation).

    Calibrates the distance-decay exponent of ``F_od = k M_o M_d d_od^(-beta)``
    by Poisson maximum likelihood with the mass product as offset. Its score
    equations make the model reproduce the total flow and the mean log trip
    distance ``sum F log d / sum F`` -- the classical calibration condition
    (Hyman 1969; Fotheringham and O'Kelly 1989, ch. 3) -- and the estimate is
    ``glm(F ~ log(d) + offset(log(M_o M_d)), family = poisson)``.

    Parameters
    ----------
    flows : array-like, shape (n,)
        Observed flows of the n origin-destination pairs.
    mass_o, mass_d : array-like, shape (n,)
        Positive origin and destination masses of each pair.
    dist : array-like, shape (n,)
        Positive distances.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``beta``; ``extra`` has ``k``, ``se_beta``,
        ``fitted`` and ``loglik``.

    References
    ----------
    Hyman, G. M. (1969). The calibration of trip distribution models. *Environment and Planning*,
    1(1), 105-112.

    Fotheringham, A. S. and O'Kelly, M. E. (1989). *Spatial Interaction Models: Formulations and
    Applications*. Kluwer, Dordrecht.

    Examples
    --------
    >>> F = [12.0, 3.0, 30.0, 7.0, 55.0, 4.0, 9.0, 21.0]
    >>> mo = [5, 5, 9, 9, 20, 20, 7, 7]
    >>> md = [9, 20, 5, 20, 5, 9, 20, 9]
    >>> d = [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.5, 1.5]
    >>> round(igravcl(F, mo, md, d).statistic, 10)
    0.9079358251
    """
    F = gv.vec(flows)
    X = gv.design(mass_o, mass_d, dist)
    Z = [[1.0, r[3]] for r in X]
    off = [r[1] + r[2] for r in X]
    g = gv.glm_fit(Z, F, "poisson", offset=off)
    b0, b1 = g["coefficients"]
    return SpatialResult(
        name="igravcl",
        statistic=-b1,
        extra={"k": math.exp(b0), "se_beta": g["se"][1], "fitted": g["fitted"], "loglik": g["loglik"]},
    )


igravcl_fn = igravcl


def cheatsheet() -> str:
    return "igravcl(flows, mass_o, mass_d, dist) -> distance-decay beta by Poisson ML"
