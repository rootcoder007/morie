# morie.fn -- function file (rootcoder007/morie)
r"""Polynomial and kernel marginal structural models.

Hernan, Brumback & Robins (2002) fit a marginal structural model for a
*repeated-measures* outcome under a time-varying treatment, and the
part this module implements is their treatment of the exposure summary:
the MSM regresses the outcome on a flexible function of cumulative
treatment rather than on treatment linearly, because the effect of
staying on therapy is not assumed to accumulate at a constant rate.

Two bases, both here.

**Polynomial.** The MSM is

.. math:: E\left[Y^{\bar a}\right] = \beta_0 + \sum_{d=1}^{D}
          \beta_d \left(\textstyle\sum_k a_k\right)^{d},

fitted by weighted least squares in the pseudo-population created by
the Sec. 21.2 stabilized weights. Degree 1 is the linear MSM and the
higher degrees are what "nonlinear effects" means here.

**Kernel.** The same regression run through a radial basis expansion of
cumulative exposure, with the centres placed at the exposure quantiles
so they follow the data rather than a grid. Useful when the shape is
not polynomial -- a threshold or a plateau -- which a low-degree
polynomial cannot represent and a high-degree one represents by
oscillating.

The two are reported together, along with the fitted curve, because the
choice between them is a modelling decision and printing a single
coefficient would conceal it.

**The exposure summary is a modelling assumption, not a summary.**
Collapsing a treatment history to its total says the order of treatment
does not matter. That is exactly the assumption Hernan, Brumback &
Robins make for cumulative zidovudine, and it is stated here rather
than left implicit; ``summary="final"`` and ``summary="duration"`` are
provided for the cases where it does not hold.

References
----------
Hernan, M. A., Brumback, B. & Robins, J. M. (2002) "Estimating the
causal effect of zidovudine on CD4 count with a marginal structural
model for repeated measures", *Statistics in Medicine* 21(12),
1689-1709, doi:10.1002/sim.1144.

Robins, J. M., Hernan, M. A. & Brumback, B. (2000) "Marginal structural
models and causal inference in epidemiology", *Epidemiology* 11(5),
550-560, doi:10.1097/00001648-200009000-00011 -- the weights.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
Chapman & Hall/CRC, Sec. 21.2.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["polynomial_kernel_msm", "exposure_summary", "rbf_basis"]

_BASES = ("polynomial", "kernel", "both")
_SUMMARIES = ("cumulative", "final", "duration")


def exposure_summary(A_history, how="cumulative"):
    """Reduce a treatment history to the scalar the MSM regresses on.

    "cumulative" is the total dose, "final" the last value, "duration"
    the number of periods with a non-zero treatment. Each is a claim
    about what part of the history matters.
    """
    if how not in _SUMMARIES:
        raise ValueError("exposure_summary: how must be one of %r, got %r"
                         % (_SUMMARIES, how))
    cols = [k.vec(a) for a in A_history]
    n = len(cols[0])
    for j, c in enumerate(cols):
        if len(c) != n:
            raise ValueError("exposure_summary: time 0 has %d rows but "
                             "time %d has %d" % (n, j, len(c)))
    if how == "cumulative":
        return [sum(c[i] for c in cols) for i in range(n)]
    if how == "final":
        return [cols[-1][i] for i in range(n)]
    return [float(sum(1 for c in cols if c[i] != 0.0)) for i in range(n)]


def rbf_basis(x, n_centres=5, width=None):
    """Radial basis expansion with centres at the data quantiles.

    Placing centres on quantiles rather than on an even grid keeps them
    where the exposure actually is; a grid wastes basis functions on
    regions with no observations and leaves the dense region
    underserved. The default width is the mean spacing between centres,
    so neighbouring bases overlap by design.
    """
    xs = [float(v) for v in k.vec(x)]
    m = int(n_centres)
    if m < 1:
        raise ValueError("rbf_basis: need at least one centre, got %r"
                         % (n_centres,))
    centres = [k.quantile7(xs, (j + 0.5) / m) for j in range(m)]
    uniq = sorted(set(centres))
    if len(uniq) < 2:
        raise ValueError(
            "rbf_basis: the exposure takes one distinct value at the "
            "requested quantiles, so no basis can be built")
    if width is None:
        gaps = [uniq[j + 1] - uniq[j] for j in range(len(uniq) - 1)]
        width = sum(gaps) / len(gaps)
    h = float(width)
    if h <= 0.0:
        raise ValueError("rbf_basis: width must be positive, got %r"
                         % (width,))
    return [[math.exp(-0.5 * ((v - c) / h) ** 2) for c in centres]
            for v in xs], centres, h


def polynomial_kernel_msm(y, A_history, H_history, degree=2,
                          basis="both", summary="cumulative",
                          n_centres=5, width=None, kind="binary",
                          stabilize=True, trim=None, grid=None):
    r"""MSM with a flexible function of cumulative exposure.

    Parameters
    ----------
    y : array-like
        Outcome at end of follow-up.
    A_history : list of array-like
        Treatment at each time point.
    H_history : list of array-like
        Time-varying covariates at each time point, for the weights.
    degree : int
        Polynomial degree. 1 is the linear MSM.
    basis : {"both", "polynomial", "kernel"}
    summary : {"cumulative", "final", "duration"}
    grid : array-like, optional
        Exposure values at which to report the fitted curves.

    Returns
    -------
    RichResult
        ``estimate`` is the linear coefficient of the polynomial MSM,
        so that degree 1 reproduces the ordinary MSM slope exactly.

    Examples
    --------
    Three periods, a quadratic dose-response::

        r = polynomial_kernel_msm(y, [A0, A1, A2], [L0, L1, L2],
                                  degree=2)
        r["estimate"], r["curve_polynomial"]
    """
    if basis not in _BASES:
        raise ValueError("polynomial_kernel_msm: basis must be one of %r, "
                         "got %r" % (_BASES, basis))
    deg = int(degree)
    if deg < 1:
        raise ValueError("polynomial_kernel_msm: degree must be at least "
                         "1, got %r" % (degree,))
    A_hist = list(A_history)
    L_hist = list(H_history) if H_history is not None else \
        [None] * len(A_hist)
    if len(L_hist) != len(A_hist):
        raise ValueError(
            "polynomial_kernel_msm: %d treatment times but %d covariate "
            "blocks" % (len(A_hist), len(L_hist)))
    yv = k.vec(y)
    n = len(yv)

    w, per_time = k.ip_weights_history(A_hist, L_hist, kind=kind,
                                       stabilize=stabilize, trim=trim)
    e = exposure_summary(A_hist, summary)
    if grid is None:
        lo, hi = min(e), max(e)
        grid = [lo + (hi - lo) * t / 20.0 for t in range(21)] \
            if hi > lo else [lo]
    grid = [float(v) for v in k.vec(grid)]

    out = {"exposure": e, "weights": w, "grid": grid,
           "n": n, "n_times": len(A_hist), "degree": deg,
           "summary": summary, "basis": basis,
           "mean_weight": sum(w) / n, "max_weight": max(w),
           "per_time_mean_weight": [sum(p["weight"]) / n
                                    for p in per_time]}

    if basis in ("polynomial", "both"):
        Xp = [[e[i] ** d for d in range(1, deg + 1)] for i in range(n)]
        fp = k.wls(Xp, yv, w)
        bp = fp["coef"]
        out["coef_polynomial"] = bp
        out["se_polynomial"] = fp["se"]
        out["vcov_polynomial"] = fp["vcov"]
        out["curve_polynomial"] = [
            bp[0] + sum(bp[d] * (g ** d) for d in range(1, deg + 1))
            for g in grid]
        out["estimate"] = bp[1]
        out["se"] = fp["se"][1]

    if basis in ("kernel", "both"):
        Xk, centres, h = rbf_basis(e, n_centres=n_centres, width=width)
        fk = k.wls(Xk, yv, w)
        bk = fk["coef"]
        out["coef_kernel"] = bk
        out["se_kernel"] = fk["se"]
        out["centres"] = centres
        out["width"] = h
        out["curve_kernel"] = [
            bk[0] + sum(bk[j + 1] * math.exp(-0.5 * ((g - centres[j]) / h)
                                             ** 2)
                        for j in range(len(centres)))
            for g in grid]
        if basis == "kernel":
            # no single slope exists, so report the average derivative
            slopes = [(out["curve_kernel"][t + 1] - out["curve_kernel"][t])
                      / (grid[t + 1] - grid[t])
                      for t in range(len(grid) - 1)
                      if grid[t + 1] != grid[t]]
            out["estimate"] = (sum(slopes) / len(slopes) if slopes
                               else float("nan"))
            out["se"] = float("nan")

    out["method"] = ("marginal structural model with a %s exposure basis, "
                     "Hernan, Brumback & Robins (2002); weights by "
                     "Robins, Hernan & Brumback (2000)" % basis)
    return RichResult(payload=out)


def cheatsheet():
    return ("polkrn: MSM on a flexible function of cumulative exposure "
            "(Hernan-Brumback-Robins 2002). polynomial degree D or RBF "
            "with quantile centres; weights are the Sec.21.2 product. "
            "summary = cumulative | final | duration.")


# compact alias per ledger/NAMING.md
polynomialkernelmsm = polynomial_kernel_msm
