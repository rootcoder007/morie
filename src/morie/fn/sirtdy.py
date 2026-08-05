# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Age-structured SIR driven by a social contact matrix."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sir_age_structured"]


def sir_age_structured(S, I, R, contact_matrix, gamma, t_max=160.0, dt=0.1):
    r"""Integrate an SIR model stratified into age groups, with mixing
    supplied by a POLYMOD-style contact matrix.

    For group *i* with :math:`N_i = S_i + I_i + R_i`,

    .. math::

        \frac{dS_i}{dt} = -S_i \sum_j C_{ij}\,\frac{I_j}{N_j}, \qquad
        \frac{dI_i}{dt} = S_i \sum_j C_{ij}\,\frac{I_j}{N_j}
                          - \gamma I_i, \qquad
        \frac{dR_i}{dt} = \gamma I_i,

    where :math:`C_{ij}` is the effective transmission rate from an
    infectious member of group *j* to a susceptible in group *i*, that is,
    the mean number of contacts per unit time scaled by the per-contact
    transmission probability.  Mossong et al. report the contact counts;
    multiplying them by the transmission probability gives :math:`C`.

    The basic reproduction number is the spectral radius of the
    next-generation matrix

    .. math::  K_{ij} = \frac{S_i(0)\,C_{ij}}{\gamma\,N_j},

    computed here by power iteration from a uniform start vector, which is
    deterministic and so identical across language arms.  With one age group
    this collapses to the scalar :math:`R_0 = C\,S_0/(\gamma N)`.

    Parameters
    ----------
    S, I, R : array-like
        Initial counts per age group; all three must have the same length.
    contact_matrix : array-like
        Square matrix :math:`C`, one row/column per age group.
    gamma : float
        Recovery rate, common to all groups.
    t_max : float, default 160.0
        Integration horizon.
    dt : float, default 0.1
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``estimate`` is the overall attack rate, total R at ``t_max``
        divided by the total population.

    References
    ----------
    Mossong, J. et al. (2008). Social contacts and mixing patterns relevant
    to the spread of infectious diseases. PLoS Medicine 5(3), e74.
    doi:10.1371/journal.pmed.0050074
    """
    s = [float(v) for v in np.atleast_1d(np.asarray(S, dtype=float)).tolist()]
    i0 = [float(v) for v in np.atleast_1d(np.asarray(I, dtype=float)).tolist()]
    r0 = [float(v) for v in np.atleast_1d(np.asarray(R, dtype=float)).tolist()]
    m = len(s)
    if m == 0 or len(i0) != m or len(r0) != m:
        raise ValueError("sir_age_structured: S, I and R must have the same non-zero length")
    C = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(contact_matrix, dtype=float)).tolist()]
    if len(C) != m or any(len(row) != m for row in C):
        raise ValueError("sir_age_structured: contact_matrix must be m x m for m age groups")
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("sir_age_structured: gamma must be non-negative")
    if any(v < 0.0 for v in s + i0 + r0):
        raise ValueError("sir_age_structured: compartment sizes must be non-negative")
    t_max = float(t_max)
    dt = float(dt)
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("sir_age_structured: need dt > 0 and t_max >= 0")

    N = [s[k] + i0[k] + r0[k] for k in range(m)]
    if any(v <= 0.0 for v in N):
        raise ValueError("sir_age_structured: every age group must have positive size")

    S0 = list(s)
    sv, iv, rv = list(s), list(i0), list(r0)

    def deriv(a, b, c):
        da = [0.0] * m
        db = [0.0] * m
        dc = [0.0] * m
        for k in range(m):
            lam = 0.0
            for j in range(m):
                lam += C[k][j] * b[j] / N[j]
            f = a[k] * lam
            da[k] = -f
            db[k] = f - gamma * b[k]
            dc[k] = gamma * b[k]
        return da, db, dc

    nsteps = int(round(t_max / dt))
    Ntot = sum(N)
    peak_I = sum(iv) / Ntot
    peak_time = 0.0
    for step in range(nsteps):
        a1, b1, c1 = deriv(sv, iv, rv)
        a2, b2, c2 = deriv(
            [sv[k] + 0.5 * dt * a1[k] for k in range(m)],
            [iv[k] + 0.5 * dt * b1[k] for k in range(m)],
            [rv[k] + 0.5 * dt * c1[k] for k in range(m)],
        )
        a3, b3, c3 = deriv(
            [sv[k] + 0.5 * dt * a2[k] for k in range(m)],
            [iv[k] + 0.5 * dt * b2[k] for k in range(m)],
            [rv[k] + 0.5 * dt * c2[k] for k in range(m)],
        )
        a4, b4, c4 = deriv(
            [sv[k] + dt * a3[k] for k in range(m)],
            [iv[k] + dt * b3[k] for k in range(m)],
            [rv[k] + dt * c3[k] for k in range(m)],
        )
        sv = [sv[k] + (dt / 6.0) * (a1[k] + 2.0 * a2[k] + 2.0 * a3[k] + a4[k]) for k in range(m)]
        iv = [iv[k] + (dt / 6.0) * (b1[k] + 2.0 * b2[k] + 2.0 * b3[k] + b4[k]) for k in range(m)]
        rv = [rv[k] + (dt / 6.0) * (c1[k] + 2.0 * c2[k] + 2.0 * c3[k] + c4[k]) for k in range(m)]
        cur = sum(iv) / Ntot
        if cur > peak_I:
            peak_I = cur
            peak_time = (step + 1) * dt

    # next-generation matrix and its spectral radius (power iteration)
    K = [[S0[a] * C[a][b] / (gamma * N[b]) if gamma > 0.0 else float("inf") for b in range(m)] for a in range(m)]
    if gamma > 0.0:
        v = [1.0 / m] * m
        lam = 0.0
        for _ in range(2000):
            w = [sum(K[a][b] * v[b] for b in range(m)) for a in range(m)]
            nrm = sum(abs(x) for x in w)
            if nrm <= 0.0:
                lam = 0.0
                break
            v = [x / nrm for x in w]
            lam = nrm
    else:
        lam = float("inf")

    return RichResult(
        payload={
            "estimate": sum(rv) / Ntot,
            "S": sv,
            "I": iv,
            "R": rv,
            "N": N,
            "attack_rate": sum(rv) / Ntot,
            "R0": lam,
            "peak_prevalence": peak_I,
            "peak_time": peak_time,
            "n_groups": m,
            "population": Ntot,
            "conservation_error": max(abs(sv[k] + iv[k] + rv[k] - N[k]) for k in range(m)),
            "gamma": gamma,
            "t_max": t_max,
            "dt": dt,
            "method": "Age-structured SIR with contact matrix (Mossong et al. 2008)",
        }
    )


def cheatsheet():
    return "sirtdy: age-structured SIR with a POLYMOD-style contact matrix"


# compact alias per ledger/NAMING.md
siragestructured = sir_age_structured
