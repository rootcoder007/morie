# morie.fn -- function file (rootcoder007/morie)
"""Non-Gaussian random fields from Gaussian ones: log-normal, chi-square, Student t, gamma anamorphosis, binary and
truncated pluri-Gaussian categories, Poisson and Cox counts, white noise, mixtures, Schlather max-stable fields,
and geometric anisotropy of coordinates."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._rrng_core import pnorm, qgamma, qnorm, qpois
from .zschl import chol_sim

__all__ = ["transformed_field", "max_stable_field", "anisotropic_coords"]


def _unit(model):
    """The model rescaled to unit total sill (a list of nested structures or one dict)."""
    parts = model if isinstance(model, list) else [model]
    tot = ssum(float(p.get("psill", 0.0)) + float(p.get("nugget", 0.0)) for p in parts)
    out = []
    for p in parts:
        q = dict(p)
        q["psill"] = float(p.get("psill", 0.0)) / tot
        if "nugget" in q:
            q["nugget"] = float(q["nugget"]) / tot
        out.append(q)
    return out if isinstance(model, list) else out[0]


def transformed_field(
    coords,
    model,
    kind: str,
    *,
    seed: int = 1,
    nsim: int = 1,
    mean: float = 0.0,
    sd: float = 1.0,
    df: int = 3,
    threshold: float = 0.0,
    proportions=None,
    shape: float = 2.0,
    rate: float = 1.0,
    scale: float = 1.0,
    weights=None,
    models=None,
) -> RichResult:
    r"""Simulate non-Gaussian fields as pointwise transforms of standard Gaussian fields ``Z`` with correlation ``model``.

    ``Z`` is :func:`morie.fn.zschl.chol_sim` of the model rescaled to unit
    sill (realisation ``s`` of independent field ``k`` uses Philox stream ``s``
    of seed ``seed + 7919 k``). ``kind``:

    - ``lognormal``: ``exp(mean + sd Z)``;
    - ``chi2``: ``sum_{k=1}^{df} Z_k^2`` over ``df`` independent fields;
    - ``student_t``: ``Z_0 / sqrt(chi2_df / df)`` (Worsley 1994 t-field);
    - ``gamma``: anamorphosis ``F_Gamma^{-1}(Phi(Z); shape, rate)``;
    - ``binary``: ``1{Z > threshold}``;
    - ``categorical``: truncated Gaussian (Matheron et al. 1987): class
      ``k`` (0-based) where ``Phi^{-1}`` of the cumulative ``proportions``
      brackets ``Z``;
    - ``poisson``: counts ``Poisson(scale exp(mean + sd Z))`` by inversion of
      Philox uniforms (stream ``s + 500``) -- a log-Gaussian Cox count field;
    - ``cox_intensity``: the intensity ``scale exp(mean + sd Z)`` itself;
    - ``white``: independent ``N(mean, sd^2)`` values (no correlation);
    - ``mixture``: ``sum_k sqrt(w_k) Z_k`` of independent unit fields with
      correlation ``models[k]`` and weights summing to 1 (a Gaussian field with
      the weighted-average correlation).

    References
    ----------
    Worsley, K. J. (1994). Local maxima and the expected Euler
    characteristic of excursion sets of chi-squared, F and t fields.
    *Advances in Applied Probability*, 26(1), 13-42.
    Matheron, G., Beucher, H., de Fouquet, C., Galli, A., Guerillot, D. and
    Ravenne, C. (1987). Conditional simulation of the geometry of
    fluvio-deltaic reservoirs. SPE 16753.
    Moller, J., Syversveen, A. R. and Waagepetersen, R. P. (1998). Log
    Gaussian Cox processes. *Scandinavian Journal of Statistics*, 25(3),
    451-482.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> r = transformed_field([(0.0, 0.0), (1.0, 0.0)], m, "binary", seed=3)
    >>> r.field[0]
    [1.0, 0.0]
    """
    P = [tuple(float(v) for v in p) for p in coords]
    n = len(P)
    unit = _unit(model)

    def gauss(k, mdl=unit):
        return chol_sim(P, mdl, nsim=nsim, seed=seed + 7919 * k, mean=0.0)["simulations"]

    out = []
    if kind == "white":
        for s in range(nsim):
            e = random_normal(n, seed=seed, stream=s)
            out.append([mean + sd * float(v) for v in e])
        return RichResult(payload={"field": out, "kind": kind})
    Z = gauss(0)
    if kind == "lognormal":
        out = [[math.exp(mean + sd * z) for z in r] for r in Z]
    elif kind == "cox_intensity":
        out = [[scale * math.exp(mean + sd * z) for z in r] for r in Z]
    elif kind == "binary":
        out = [[1.0 if z > threshold else 0.0 for z in r] for r in Z]
    elif kind == "gamma":
        out = [[float(qgamma(float(pnorm(z)), shape, rate)) for z in r] for r in Z]
    elif kind == "categorical":
        pr = [float(v) for v in proportions]
        if abs(ssum(pr) - 1) > 1e-12 or any(v <= 0 for v in pr):
            raise ValueError("proportions must be positive and sum to 1")
        cuts = [float(qnorm(ssum(pr[: k + 1]))) for k in range(len(pr) - 1)]
        out = [[float(sum(1 for c in cuts if z > c)) for z in r] for r in Z]
    elif kind in ("chi2", "student_t"):
        fields = [Z] + [gauss(k) for k in range(1, df + (1 if kind == "student_t" else 0))]
        if kind == "chi2":
            out = [[ssum(fields[k][s][i] ** 2 for k in range(df)) for i in range(n)] for s in range(nsim)]
        else:
            out = [
                [
                    fields[0][s][i] / math.sqrt(ssum(fields[k][s][i] ** 2 for k in range(1, df + 1)) / df)
                    for i in range(n)
                ]
                for s in range(nsim)
            ]
    elif kind == "poisson":
        for s, r in enumerate(Z):
            u = random_uniform(n, seed=seed, stream=500 + s)
            out.append([float(qpois(float(ui), scale * math.exp(mean + sd * z))) for ui, z in zip(u, r)])
    elif kind == "mixture":
        w = [float(v) for v in weights]
        if abs(ssum(w) - 1) > 1e-12 or any(v < 0 for v in w):
            raise ValueError("weights must be nonnegative and sum to 1")
        fields = [gauss(k, _unit(models[k])) for k in range(len(models))]
        out = [[ssum(math.sqrt(w[k]) * fields[k][s][i] for k in range(len(w))) for i in range(n)] for s in range(nsim)]
    else:
        raise ValueError("unknown kind")
    return RichResult(payload={"field": out, "kind": kind})


def max_stable_field(coords, model, *, n_fields: int = 200, seed: int = 1) -> RichResult:
    r"""Schlather (2002) extremal Gaussian max-stable field with unit Frechet margins.

    ``Z(s) = sqrt(2 pi) max_i zeta_i max(0, W_i(s))`` with ``zeta_i = 1 /
    (E_1 + ... + E_i)`` the points of a unit-rate Poisson process on
    ``(0, inf)`` in decreasing order (``E_j`` exponential from Philox stream
    ``2000`` of ``seed``) and ``W_i`` independent standard Gaussian fields with
    correlation ``model`` (streams ``i``). The series is truncated at
    ``n_fields`` terms; the extremal coefficient ``theta(h) = 1 + sqrt((1 -
    rho(h))/2)`` is returned for the simulated pairs' distances as a check.

    References
    ----------
    Schlather, M. (2002). Models for stationary max-stable random fields.
    *Extremes*, 5(1), 33-44.

    Examples
    --------
    >>> r = max_stable_field([(0.0, 0.0), (0.5, 0.0)], {"model": "Exp", "psill": 1.0, "range": 1.0}, n_fields=20)
    >>> len(r.field), all(v > 0 for v in r.field)
    (2, True)
    """
    P = [tuple(float(v) for v in p) for p in coords]
    unit = _unit(model)
    W = chol_sim(P, unit, nsim=n_fields, seed=seed, mean=0.0)["simulations"]
    u = random_uniform(n_fields, seed=seed, stream=2000)
    g = 0.0
    Z = [0.0] * len(P)
    for i in range(n_fields):
        g += -math.log(float(u[i]))
        zeta = 1.0 / g
        for k in range(len(P)):
            v = math.sqrt(2 * math.pi) * zeta * max(0.0, W[i][k])
            if v > Z[k]:
                Z[k] = v
    return RichResult(payload={"field": Z, "n_fields": n_fields})


def anisotropic_coords(coords, angle: float, ratio: float):
    r"""Map coordinates so that an isotropic model applies to a geometrically anisotropic field (gstat ``anis``).

    ``angle`` is the direction of the major axis in degrees clockwise from
    north and ``ratio`` (``<= 1``) the minor/major range ratio: coordinates
    are rotated so the major axis becomes the y axis and the minor-axis
    component is divided by ``ratio`` (Isaaks and Srivastava 1989); use the
    result with any isotropic simulator or covariance.

    Examples
    --------
    >>> [tuple(round(v, 6) for v in p) for p in anisotropic_coords([(1.0, 0.0), (0.0, 1.0)], 90.0, 0.5)]
    [(0.0, 1.0), (-2.0, 0.0)]
    """
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for p in coords:
        x, y = float(p[0]), float(p[1])
        major = x * sa + y * ca
        minor = x * ca - y * sa
        out.append((minor / ratio, major))
    return out


def cheatsheet() -> str:
    return "transformed_field / max_stable_field / anisotropic_coords -> non-Gaussian random fields."
