# morie.fn -- function file (rootcoder007/morie)
"""Cox (doubly stochastic) process simulation given a realised intensity field."""

from __future__ import annotations

import math

from ._richresult import RichResult
from .coxproc import _U, _poisson


def cox_process(intensity_field, window, seed=None) -> RichResult:
    r"""Simulate a Cox process conditionally on one realisation of its random intensity.

    A Cox process is a Poisson process whose intensity ``Lambda`` is itself
    random; given ``Lambda = lambda`` it is an inhomogeneous Poisson process.
    ``intensity_field`` is taken as that realised intensity, piecewise
    constant on an ``ny x nx`` lattice over ``window = (xmin, xmax, ymin,
    ymax)`` (row ``iy`` is the ``iy``-th band from ``ymin``), so this draws the
    conditional Poisson process: each cell receives a Poisson number of points
    with mean ``max(lambda, 0) |cell|`` (inversion), placed uniformly in the
    cell, with Philox draws as :func:`morie.fn.coxproc.lgcp_simulate` (seed
    ``seed``, streams 1000+). Drawing the random intensity as well (the
    unconditional log-Gaussian Cox process) is
    :func:`morie.fn.coxproc.lgcp_simulate`; the Thomas cluster process is
    :func:`morie.fn.coxproc.thomas_simulate`.

    Parameters
    ----------
    intensity_field : ``ny x nx`` nested list of intensities.
    window : ``(xmin, xmax, ymin, ymax)``.
    seed : Philox seed (default 1).

    Returns
    -------
    RichResult
        ``points`` (list of ``(x, y)``), ``n_points``, ``value`` (the count),
        ``mean_intensity``, ``expected_count`` (``sum lambda |cell|``).

    References
    ----------
    Cox, D. R. (1955). Some statistical methods connected with series of
    events. *Journal of the Royal Statistical Society B* 17, 129-164.

    Moller, J., Syversveen, A. R. and Waagepetersen, R. P. (1998). Log
    Gaussian Cox processes. *Scandinavian Journal of Statistics* 25, 451-482.

    Examples
    --------
    >>> r = cox_process([[2.0, 5.0], [1.0, 8.0]], (0, 2, 0, 2), seed=3)
    >>> r.n_points, r.expected_count, all(0 <= x <= 2 and 0 <= y <= 2 for x, y in r.points)
    (15, 16.0, True)
    """
    lam = [[float(v) for v in row] for row in intensity_field]
    ny, nx = len(lam), len(lam[0])
    if any(len(row) != nx for row in lam):
        raise ValueError("intensity_field must be a rectangular grid")
    x0, x1, y0, y1 = [float(v) for v in window]
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    U = _U(1 if seed is None else int(seed), 1000)
    pts = []
    for iy in range(ny):
        for ix in range(nx):
            k = _poisson(max(lam[iy][ix], 0.0) * dx * dy, U)
            for _ in range(k):
                pts.append((x0 + ix * dx + dx * U.next(), y0 + iy * dy + dy * U.next()))
    tot = math.fsum(max(v, 0.0) for row in lam for v in row)
    return RichResult(
        payload={
            "name": "cox_process",
            "points": pts,
            "n_points": len(pts),
            "value": float(len(pts)),
            "mean_intensity": math.fsum(v for row in lam for v in row) / (nx * ny),
            "expected_count": tot * dx * dy,
        }
    )


sgcox = cox_process


def cheatsheet() -> str:
    return (
        "cox_process(intensity_field, window, seed) -> Cox process given its realised intensity (Poisson on the field)."
    )
