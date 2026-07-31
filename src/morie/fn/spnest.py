"""Nested covariance/variogram model: sum of valid components."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import semivariogram, _as_lag

__all__ = ["schabenberger_nested_variogram"]


def schabenberger_nested_variogram(h, components=None):
    r"""
    Nested semivariogram: the linear model of regionalization.

    With :math:`Z(s) = \mu + \sum_{j=1}^{p} a_j U_j(s)` and the
    :math:`U_j` mutually orthogonal,

    .. math::

        C_z(h) = \sum_{j=1}^{p} a_j^2 C_j(h), \qquad
        \gamma_z(h) = \sum_{j=1}^{p} a_j^2 \gamma_j(h)

    Validity relies on the components being ORTHOGONAL; the book notes
    this is hard to justify except when a white-noise measurement-error
    component is nested with one other model to create a nugget.

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    components : sequence of mapping
        Each component is ``{'model': ..., 'sill': ..., 'range': ...}``
        with an optional ``'nugget'``. ``model='nugget'`` gives a pure
        white-noise component (its ``sill`` is the nugget).

    Returns
    -------
    RichResult
        ``gamma`` (the nested total), ``components`` (per-component
        gammas) and ``total_sill``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.6, eqs.
    (4.22)-(4.23), p. 150.
    """
    h = _as_lag(h)
    if not components:
        raise ValueError("`components` must list at least one component")
    total = np.zeros_like(h)
    parts, sill_sum = [], 0.0
    for k, c in enumerate(components):
        model = c.get("model", "exponential")
        sill = float(c.get("sill", 1.0))
        if sill < 0:
            raise ValueError(f"component {k}: `sill` must be >= 0")
        if model == "nugget":
            g = np.where(h > 0, sill, 0.0)
        else:
            g = semivariogram(h, 0.0, sill, float(c.get("range", 1.0)), model)
        parts.append({"model": model, "sill": sill, "gamma": g})
        total = total + g
        sill_sum += sill
    return RichResult(
        title="Nested semivariogram (linear model of regionalization)",
        summary_lines=[("components", len(parts)), ("total sill", sill_sum)],
        payload={"gamma": total, "components": parts,
                 "total_sill": float(sill_sum)},
    )


def cheatsheet():
    return "spnest: nested semivariogram, gamma_z(h) = sum_j a_j^2 gamma_j(h)."
