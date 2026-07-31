# SPDX-License-Identifier: AGPL-3.0-or-later
"""Separable spatio-temporal covariance functions."""

import numpy as np

from ._richresult import RichResult
from ._schab_st import (is_separable, is_valid_covariance,
                        separable_covariance, st_lag_matrices)

__all__ = ["schabenberger_st_cov_separable"]


def schabenberger_st_cov_separable(spatial_h, temporal_u, cov_spatial,
                                   cov_temporal, form="product", coords=None,
                                   times=None):
    """Separable spatio-temporal covariance, Sec. 9.2.

    A separable covariance decomposes Cov[Z(s,t), Z(s+h,t+k)] into a purely
    spatial and a purely temporal component, combined by multiplication or
    addition:

        product      C(h,k) = Cs(h; theta_s) Ct(k; theta_t)
        sum          C(h,k) = Cs(h; theta_s) + Ct(k; theta_t)
        product_sum  C(h,k) = Cs Ct + Cs + Ct

    Both of the first two are valid whenever the components are, by the two
    elementary properties the text quotes at the head of Sec. 9.2: a
    non-negative combination of valid covariance functions is valid, and so
    is a product. The components normally carry DIFFERENT parameters, which
    is what accommodates space-time anisotropy -- the whole point of eq (9.3)
    and the reason Sec. 9.1 rejects treating time as a third coordinate.

    ``product_sum`` is De Cesare, Myers and Posa (2001). It appears in this
    section of the book but the text is explicit that it "is generally
    nonseparable", so the returned ``separable`` flag is False for it.

    The drawback the text identifies is reported rather than hidden: under
    product separability the spatial covariances at different time lags are
    proportional to one another, so the spatial dependence has the same SHAPE
    at every time lag. The temporal and spatial components "do not act upon
    each other". If that matters, a non-separable model is required --
    :func:`schabenberger_st_cov_nonsep`.

    Parameters
    ----------
    spatial_h, temporal_u : array-like
        Spatial lag ||h|| (non-negative) and temporal lag k. Carried
        separately throughout; they are never concatenated into one vector.
    cov_spatial, cov_temporal : callable
        Cs(h) and Ct(k). Each must be a valid covariance function in its own
        domain.
    form : {"product", "sum", "product_sum"}
    coords, times : array-like, optional
        If given, eq (9.5) is checked numerically on that design and the
        minimum eigenvalue is reported. Construction alone is not proof of
        validity: Sec. 9.3 records that Gneiting (2002) found published
        covariance functions in Cressie and Huang (1999) to be invalid.

    Returns
    -------
    RichResult
        Keys: ``st_covariance``, ``separable``, ``form``, ``spatial_only``,
        ``temporal_only``, and when a design is supplied ``valid`` and
        ``min_eigenvalue``.

    References
    ----------
    Schabenberger & Gotway (2005), Sec. 9.2, eqs (9.5)-(9.6).
    """
    c = separable_covariance(spatial_h, temporal_u, cov_spatial, cov_temporal,
                             form=form)
    h = np.atleast_1d(np.asarray(spatial_h, dtype=float))
    k = np.atleast_1d(np.asarray(temporal_u, dtype=float))
    zero = np.zeros_like(h)
    payload = {
        "st_covariance": c,
        "form": form,
        "separable": is_separable(form),
        # C(h, 0) and C(0, k), which Sec. 9.2 names as the spatial and
        # temporal covariance functions of the process
        "spatial_only": separable_covariance(h, zero, cov_spatial,
                                             cov_temporal, form=form),
        "temporal_only": separable_covariance(np.zeros_like(k), k, cov_spatial,
                                              cov_temporal, form=form),
        "sill": float(np.asarray(separable_covariance(
            np.array(0.0), np.array(0.0), cov_spatial, cov_temporal,
            form=form)).ravel()[0]),
    }
    lines = [("form", form), ("separable", payload["separable"]),
             ("C(0,0)", payload["sill"])]

    if coords is not None and times is not None:
        v = is_valid_covariance(
            coords, times,
            lambda d, u: separable_covariance(d, u, cov_spatial, cov_temporal,
                                              form=form))
        payload.update(valid=v["valid"], min_eigenvalue=v["min_eigenvalue"])
        lines += [("positive definite", v["valid"]),
                  ("min eigenvalue", v["min_eigenvalue"])]
        if not v["valid"]:
            payload["warning"] = (
                "eq (9.5) fails on this design: the construction does not "
                "yield a valid covariance function here")

    return RichResult(title="Separable spatio-temporal covariance",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spstcv: separable spatio-temporal covariance (Sec. 9.2) -- "
            "product, sum and product-sum forms, with the eq (9.5) validity "
            "check")
