# morie.fn -- function file (rootcoder007/morie)
r"""RNA velocity from the full splicing kinetics, not the steady state.

**The kinetics.** For one gene, unspliced precursor :math:`u(t)` and
spliced mature mRNA :math:`s(t)` obey

.. math:: \frac{du}{dt} = \alpha^{(k)}(t) - \beta u(t), \qquad
          \frac{ds}{dt} = \beta u(t) - \gamma s(t),

with transcription rate :math:`\alpha^{(k)}` switching between an
induction ("on") and a repression ("off") phase, splicing rate
:math:`\beta` and degradation rate :math:`\gamma`. RNA velocity *is*
:math:`ds/dt = \beta u - \gamma s`; everything else is inference of
the parameters.

**Why the steady-state model is not enough.** The original approach
reads velocity off the residual from a fitted steady-state ratio
:math:`\gamma/\beta`, which needs two assumptions: that the full
dynamics are observed for each gene, so the steady states actually
appear in the data, and that all genes share one splicing rate. Both
fail on transient populations and on mixtures of subpopulations with
different kinetics. ``steady_state_velocity`` implements that model,
because it is the baseline the paper improves on, and
``dynamical_fit`` solves the kinetics instead.

**The closed form is the point.** With :math:`\tau` the time since the
last phase switch and :math:`(u_0, s_0)` the state at the switch,

.. math:: u(\tau) = u_0 e^{-\beta\tau}
          + \frac{\alpha}{\beta}\big(1 - e^{-\beta\tau}\big),

.. math:: s(\tau) = s_0 e^{-\gamma\tau}
          + \frac{\alpha}{\gamma}\big(1 - e^{-\gamma\tau}\big)
          + \frac{\alpha - \beta u_0}{\gamma - \beta}
            \big(e^{-\gamma\tau} - e^{-\beta\tau}\big).

Solving explicitly is what lets an *unobserved* steady state still be
inferred. The anchor holds this against a Runge-Kutta integration of
the ODEs themselves, so a slip in the algebra fails rather than
propagating.

The :math:`\gamma = \beta` case is a removable singularity in that
expression, not a real one; ``solve_kinetics`` takes the limit rather
than dividing by zero.

**Inference.** Expectation-maximisation, as in the paper: in the E
step each observation :math:`x_i = (u_i, s_i)` is assigned the latent
time :math:`t_i` minimising its distance to the phase trajectory, and
a transcriptional state :math:`k_i \in \{`on, off, steady-on,
steady-off :math:`\}` by likelihood on the corresponding segment; in
the M step the rates are updated. ``dynamical_fit`` records the
likelihood at every iteration so the monotone increase is visible.

References
----------
Bergen, V., Lange, M., Peidli, S., Wolf, F. A. & Theis, F. J. (2019)
"Generalizing RNA velocity to transient cell states through dynamical
modeling", bioRxiv 820936, doi:10.1101/820936; published as Bergen et
al. (2020) *Nature Biotechnology* 38(12), 1408-1414,
doi:10.1038/s41587-020-0591-3. The splicing ODEs reproduced above, the
two assumptions the steady-state model needs (full dynamics observed
per gene, one shared splicing rate) and why transient or heterogeneous
populations violate them, the explicit solution of the kinetics, the
latent variables (a discrete state :math:`k_i` and continuous time
:math:`t_i` per cell), the EM scheme assigning :math:`t_i` by minimum
distance to the phase trajectory and :math:`k_i` by segment
likelihood, and velocity as the derivative of spliced abundance.

La Manno, G., Soldatov, R., Zeisel, A., Braun, E., Hochgerner, H.,
Petukhov, V., Lidschreiber, K., Kastriti, M. E., Lönnerberg, P.,
Furlan, A., Fan, J., Borm, L. E., Liu, Z., van Bruggen, D., Guo, J.,
He, X., Barker, R., Sundström, E., Castelo-Branco, G., Cramer, P.,
Adameyko, I., Linnarsson, S. & Kharchenko, P. V. (2018) "RNA velocity
of single cells", *Nature* 560(7719), 494-498,
doi:10.1038/s41586-018-0414-6, for the steady-state model this
generalises.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["STATES", "solve_kinetics", "velocity", "simulate_gene",
           "steady_state_velocity", "assign_latent_time",
           "dynamical_fit", "latent_time"]

STATES = ("on", "off", "steady_on", "steady_off")


def solve_kinetics(tau, alpha, beta, gamma, u0=0.0, s0=0.0):
    r"""The explicit solution of the splicing ODEs at time
    :math:`\tau` after a phase switch."""
    for name, v in (("beta", beta), ("gamma", gamma)):
        if v <= 0.0:
            raise ValueError("scvelo: %s must be positive" % name)
    if alpha < 0.0:
        raise ValueError("scvelo: the transcription rate cannot be "
                         "negative")
    t = float(tau)
    if t < 0.0:
        raise ValueError("scvelo: tau cannot be negative")
    eb = math.exp(-beta * t)
    eg = math.exp(-gamma * t)
    u = u0 * eb + (alpha / beta) * (1.0 - eb)
    if abs(gamma - beta) < 1e-10:
        # removable singularity: the limit gamma -> beta of the
        # (e^-gt - e^-bt)/(g-b) term is t e^-bt.
        s = s0 * eb + (alpha / beta) * (1.0 - eb) \
            - (alpha - beta * u0) * t * eb
    else:
        s = (s0 * eg + (alpha / gamma) * (1.0 - eg)
             + (alpha - beta * u0) / (gamma - beta) * (eg - eb))
    return {"u": u, "s": s, "tau": t}


def velocity(u, s, beta, gamma):
    r""":math:`\nu = \beta u - \gamma s`, the derivative of spliced
    abundance."""
    return beta * float(u) - gamma * float(s)


def simulate_gene(alpha, beta, gamma, t_switch, times):
    r"""A gene through induction and repression, on the closed form."""
    if t_switch < 0.0:
        raise ValueError("scvelo: t_switch cannot be negative")
    sw = solve_kinetics(t_switch, alpha, beta, gamma)
    out = []
    for t in times:
        if t <= t_switch:
            st = solve_kinetics(t, alpha, beta, gamma)
            k = "on"
        else:
            st = solve_kinetics(t - t_switch, 0.0, beta, gamma,
                                sw["u"], sw["s"])
            k = "off"
        out.append({"t": float(t), "u": st["u"], "s": st["s"],
                    "state": k,
                    "velocity": velocity(st["u"], st["s"], beta,
                                         gamma)})
    return {"observations": out, "switch": sw,
            "steady_on": {"u": alpha / beta, "s": alpha / gamma}}


def steady_state_velocity(u, s, quantile=0.95):
    r"""The baseline model: regress on the extreme quantiles.

    Fits :math:`\gamma/\beta` through the origin on the cells assumed
    to be at steady state, and calls the residual the velocity. It
    needs the steady states to be present in the data, which is the
    assumption the dynamical model removes.
    """
    n = len(u)
    if n != len(s):
        raise ValueError("scvelo: u and s must have the same length")
    if n < 3:
        raise ValueError("scvelo: need at least three cells")
    order = sorted(range(n), key=lambda i: s[i])
    keep = set(order[:max(1, int(n * (1.0 - quantile)))]) \
        | set(order[-max(1, int(n * (1.0 - quantile))):])
    num = sum(u[i] * s[i] for i in keep)
    den = sum(s[i] * s[i] for i in keep)
    if den <= 0.0:
        raise ValueError("scvelo: the spliced counts are all zero at "
                         "the fitted extremes")
    ratio = num / den
    return {"gamma_over_beta": ratio,
            "velocity": [u[i] - ratio * s[i] for i in range(n)],
            "n_fitted": len(keep),
            "assumptions": "steady states observed, and one splicing "
                           "rate shared across genes",
            "method": "steady-state model; La Manno et al. (2018)"}


def assign_latent_time(u, s, alpha, beta, gamma, t_switch,
                       grid=200, t_max=None):
    r"""E step: the time on the trajectory closest to each
    observation."""
    if t_max is None:
        t_max = 2.0 * t_switch + 5.0 / min(beta, gamma)
    ts = [t_max * k / float(grid) for k in range(grid + 1)]
    traj = simulate_gene(alpha, beta, gamma, t_switch,
                         ts)["observations"]
    out = []
    for i in range(len(u)):
        best = None
        for p in traj:
            d = (p["u"] - u[i]) ** 2 + (p["s"] - s[i]) ** 2
            if best is None or d < best[0]:
                best = (d, p)
        out.append({"t": best[1]["t"], "state": best[1]["state"],
                    "distance": math.sqrt(best[0])})
    return out


def _residual(u, s, alpha, beta, gamma, t_switch, grid=200):
    a = assign_latent_time(u, s, alpha, beta, gamma, t_switch, grid)
    return sum(x["distance"] ** 2 for x in a), a


def dynamical_fit(u, s, alpha0=None, beta0=1.0, gamma0=0.5,
                  t_switch0=None, n_iter=25, grid=120):
    r"""EM over the rates and the latent variables."""
    n = len(u)
    if n != len(s):
        raise ValueError("scvelo: u and s must have the same length")
    if n < 4:
        raise ValueError("scvelo: need at least four cells")
    alpha = max(u) * beta0 if alpha0 is None else float(alpha0)
    beta, gamma = float(beta0), float(gamma0)
    t_switch = (1.0 / beta if t_switch0 is None else float(t_switch0))
    history = []
    for _ in range(int(n_iter)):
        rss, assign = _residual(u, s, alpha, beta, gamma, t_switch,
                                grid)
        history.append(rss)
        best = (rss, alpha, beta, gamma, t_switch)
        for scale in (0.8, 0.9, 1.1, 1.25):
            for which in range(4):
                cand = [alpha, beta, gamma, t_switch]
                cand[which] *= scale
                if min(cand[1], cand[2]) <= 0.0 or cand[0] < 0.0:
                    continue
                r2, _ = _residual(u, s, cand[0], cand[1], cand[2],
                                  cand[3], grid)
                if r2 < best[0]:
                    best = (r2, cand[0], cand[1], cand[2], cand[3])
        if best[0] >= rss - 1e-12:
            break
        _, alpha, beta, gamma, t_switch = best
    rss, assign = _residual(u, s, alpha, beta, gamma, t_switch, grid)
    return RichResult(payload={
        "estimate": rss, "alpha": alpha, "beta": beta, "gamma": gamma,
        "t_switch": t_switch, "rss": rss, "rss_history": history,
        "latent": assign,
        "velocity": [velocity(u[i], s[i], beta, gamma)
                     for i in range(n)],
        "steady_on": {"u": alpha / beta, "s": alpha / gamma},
        "method": "dynamical model by EM on the explicit kinetics; "
                  "Bergen et al. (2019)",
    })


def latent_time(fits):
    r"""A gene-shared clock: the per-cell median of the gene times."""
    if not fits:
        raise ValueError("scvelo: no gene fits supplied")
    n = len(fits[0]["latent"])
    if any(len(f["latent"]) != n for f in fits):
        raise ValueError("scvelo: every gene must cover the same "
                         "cells")
    out = []
    for i in range(n):
        ts = sorted(f["latent"][i]["t"] for f in fits)
        m = len(ts)
        out.append(ts[m // 2] if m % 2 else
                   0.5 * (ts[m // 2 - 1] + ts[m // 2]))
    return {"latent_time": out, "n_genes": len(fits), "n_cells": n,
            "note": "gene times coupled into one clock so rates are "
                    "comparable across genes"}


def cheatsheet():
    return ("scvelo: du/dt = alpha - beta u, ds/dt = beta u - gamma s, "
            "and velocity IS ds/dt. The steady-state model reads "
            "velocity off a fitted gamma/beta ratio and needs the "
            "steady states to be observed and one splicing rate "
            "shared; the dynamical model solves the kinetics in "
            "closed form and infers rates plus a per-cell latent time "
            "and state by EM, so unobserved steady states are still "
            "recovered. gamma = beta is a removable singularity, "
            "taken as a limit rather than a division by zero.")


# compact alias per ledger/NAMING.md
rna_velocity = dynamical_fit


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
rnavelocity = rna_velocity
