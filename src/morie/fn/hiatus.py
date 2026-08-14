# morie.fn -- function file (rootcoder007/morie)
r"""Many-strain pathogen dynamics without the combinatorial blow-up.

A host who has met :math:`n` antigenically distinct strains has
:math:`2^n` possible immune histories. Track those and ten strains
already needs 1,024 variables; a hundred is hopeless. That single fact
is why multi-strain models were stuck at three or four strains.

**The status-based move.** Take the pathogen's view rather than the
host's: instead of asking what a host has *been infected by*, ask only
what the host's current state implies for *future infections by each
strain*. Combined with two further choices -- immunity acting on
transmission rather than susceptibility, and **polarized** immunity, in
which partial cross-protection means some hosts become fully immune
rather than all hosts becoming partly immune -- every host needs exactly
one variable per strain. The system is

.. math:: \dot I_i &= \beta_i S_i I_i - \nu_i I_i - \mu I_i, \\
          \dot S_i &= \mu - \sum_j \beta_j S_i \sigma_{ij} I_j
                      - \mu S_i,

for :math:`i = 1, \dots, n`, where :math:`S_i` and :math:`I_i` are the
numbers susceptible to and infectious with strain :math:`i`,
:math:`\beta_i` and :math:`\nu_i` its transmission and recovery rates,
:math:`\mu` the birth and death rate at demographic equilibrium, and
:math:`\sigma_{ij}` the chance that an infection by strain :math:`j`
confers immunity to strain :math:`i`.

**Complexity is now linear.** :math:`2n` variables, not :math:`2^n`.
The cross-immunity structure is an arbitrary matrix -- nothing is
assumed about it beyond being a matrix -- which is what lets the model
carry a hundred strains, or a continuum.

**What the elegance costs.** The property that makes it work is that
every host has the same chance of gaining immunity to a strain
regardless of their current immune state. That is an abstraction, not a
fact about immune systems, and it is the one to challenge before
trusting a result.

**The structure the anchor exploits.** Two limits pin the model down
exactly:

* :math:`\sigma_{ij} = 0` for :math:`i \ne j` -- strains do not
  interact at all, so each must reach its own single-strain endemic
  equilibrium, :math:`S_i^{*} = (\nu_i+\mu)/\beta_i = 1/R_{0i}` and
  :math:`I_i^{*} = \mu(1 - 1/R_{0i})/(\nu_i+\mu)`. Those are closed
  forms, and the integrator must land on them.
* :math:`\sigma_{ij} = 1` everywhere -- immunity to one is immunity to
  all, so the strains compete for one susceptible pool and only the
  largest :math:`R_0` survives. Competitive exclusion, and the model
  must produce it rather than be told it.

**Strain space.** Applications need a geometry: which strains are close
enough to cross-react, and which are a mutation apart. The simplest is
a line, with mutation to immediate neighbours and cross-immunity that
falls off with distance -- the arrangement of the paper's Fig. 1, built
here by ``linear_strain_space``.

References
----------
Gog, J. R. & Grenfell, B. T. (2002) "Dynamics and selection of
many-strain pathogens", *Proceedings of the National Academy of
Sciences* 99(26), 17209-17214, doi:10.1073/pnas.252512799. "Derivation
of the model" (the status-based formulation, polarized immunity,
reduced transmission, and the two differential equations implemented
here) and the following section (linear strain space with stepwise
mutation, Fig. 1).

Gog, J. R. & Swinton, J. (2002) "A status-based approach to multiple
strain dynamics", *Journal of Mathematical Biology* 44(2), 169-184,
doi:10.1007/s002850100120. Reference 7 of the above; the status-based
framework the paper refines.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["linear_strain_space", "derivatives", "simulate",
           "endemic_equilibrium", "basic_reproduction_numbers"]

_EPS = 1e-14


def _check(beta, nu, mu, sigma):
    b = [float(v) for v in k.vec(beta)]
    n = len(b)
    nv = ([float(nu)] * n if isinstance(nu, (int, float))
          else [float(v) for v in k.vec(nu)])
    if len(nv) != n:
        raise ValueError("hiatus: %d recovery rates for %d strains"
                         % (len(nv), n))
    if n < 1:
        raise ValueError("hiatus: at least one strain is needed")
    if any(v <= 0.0 for v in b):
        raise ValueError("hiatus: transmission rates must be positive")
    if any(v < 0.0 for v in nv):
        raise ValueError("hiatus: recovery rates must be "
                         "non-negative")
    if float(mu) < 0.0:
        raise ValueError("hiatus: the birth/death rate must be "
                         "non-negative")
    S = k.mat(sigma)
    if len(S) != n or any(len(r) != n for r in S):
        raise ValueError("hiatus: sigma must be %d by %d" % (n, n))
    Sm = [[float(v) for v in r] for r in S]
    for r in Sm:
        for v in r:
            if not 0.0 <= v <= 1.0:
                raise ValueError("hiatus: sigma entries are "
                                 "probabilities and must lie in "
                                 "[0, 1], got %r" % (v,))
    return b, nv, float(mu), Sm, n


def basic_reproduction_numbers(beta, nu, mu):
    r""":math:`R_{0i} = \beta_i / (\nu_i + \mu)`, strain by strain."""
    b = [float(v) for v in k.vec(beta)]
    n = len(b)
    nv = ([float(nu)] * n if isinstance(nu, (int, float))
          else [float(v) for v in k.vec(nu)])
    m = float(mu)
    out = []
    for i in range(n):
        d = nv[i] + m
        if d <= _EPS:
            raise ValueError("hiatus: strain %d never leaves the "
                             "infectious class (nu + mu = 0)" % i)
        out.append(b[i] / d)
    return out


def endemic_equilibrium(beta, nu, mu, strain=0):
    r"""Single-strain closed form, for a strain in isolation.

    :math:`S^{*} = 1/R_0`, :math:`I^{*} = \mu(1 - 1/R_0)/(\nu+\mu)`.
    Valid when :math:`\sigma_{ij} = 0` off the diagonal, which is
    exactly when the strains decouple.
    """
    R0 = basic_reproduction_numbers(beta, nu, mu)[int(strain)]
    if R0 <= 1.0:
        return {"R0": R0, "S": 1.0, "I": 0.0,
                "note": "R0 <= 1, so the disease-free state is the "
                        "only equilibrium"}
    b = [float(v) for v in k.vec(beta)]
    n = len(b)
    nv = ([float(nu)] * n if isinstance(nu, (int, float))
          else [float(v) for v in k.vec(nu)])
    d = nv[int(strain)] + float(mu)
    return {"R0": R0, "S": 1.0 / R0,
            "I": float(mu) * (1.0 - 1.0 / R0) / d}


def derivatives(S, I, beta, nu, mu, sigma):
    r"""The right-hand sides of the two equations, as printed."""
    b, nv, m, sg, n = _check(beta, nu, mu, sigma)
    Sv = [float(v) for v in k.vec(S)]
    Iv = [float(v) for v in k.vec(I)]
    if len(Sv) != n or len(Iv) != n:
        raise ValueError("hiatus: S and I must have one entry per "
                         "strain (%d, %d, %d)" % (len(Sv), len(Iv), n))
    dI = [b[i] * Sv[i] * Iv[i] - nv[i] * Iv[i] - m * Iv[i]
          for i in range(n)]
    dS = [m - sum(b[j] * Sv[i] * sg[i][j] * Iv[j] for j in range(n))
          - m * Sv[i] for i in range(n)]
    return dS, dI


def simulate(beta, nu, mu, sigma, S0=None, I0=None, t_end=2000.0,
             dt=0.05, mutation=0.0, record_every=100):
    r"""Integrate the system by fourth-order Runge-Kutta.

    ``mutation`` adds the stepwise term of the applications section:
    each strain leaks a fraction of its incidence to its immediate
    neighbours on the strain line. Set it to 0 for the bare system.
    """
    b, nv, m, sg, n = _check(beta, nu, mu, sigma)
    Sv = [1.0] * n if S0 is None else [float(v) for v in k.vec(S0)]
    Iv = ([1e-4] * n if I0 is None
          else [float(v) for v in k.vec(I0)])
    if len(Sv) != n or len(Iv) != n:
        raise ValueError("hiatus: the initial state must have one "
                         "entry per strain")
    if float(dt) <= 0.0:
        raise ValueError("hiatus: dt must be positive")
    mu_rate = float(mutation)
    if not 0.0 <= mu_rate < 1.0:
        raise ValueError("hiatus: mutation must lie in [0, 1)")

    def rhs(Sx, Ix):
        dS, dI = derivatives(Sx, Ix, b, nv, m, sg)
        if mu_rate > 0.0 and n > 1:
            born = [b[i] * Sx[i] * Ix[i] for i in range(n)]
            leak = [mu_rate * v for v in born]
            for i in range(n):
                dI[i] -= leak[i]
                nb = [q for q in (i - 1, i + 1) if 0 <= q < n]
                for q in nb:
                    dI[q] += leak[i] / len(nb)
        return dS, dI

    steps = int(float(t_end) / float(dt))
    traj_t, traj_S, traj_I = [], [], []
    h = float(dt)
    for s in range(steps + 1):
        if s % int(record_every) == 0:
            traj_t.append(s * h)
            traj_S.append(list(Sv))
            traj_I.append(list(Iv))
        if s == steps:
            break
        k1S, k1I = rhs(Sv, Iv)
        aS = [Sv[i] + 0.5 * h * k1S[i] for i in range(n)]
        aI = [Iv[i] + 0.5 * h * k1I[i] for i in range(n)]
        k2S, k2I = rhs(aS, aI)
        bS = [Sv[i] + 0.5 * h * k2S[i] for i in range(n)]
        bI = [Iv[i] + 0.5 * h * k2I[i] for i in range(n)]
        k3S, k3I = rhs(bS, bI)
        cS = [Sv[i] + h * k3S[i] for i in range(n)]
        cI = [Iv[i] + h * k3I[i] for i in range(n)]
        k4S, k4I = rhs(cS, cI)
        for i in range(n):
            Sv[i] += h / 6.0 * (k1S[i] + 2 * k2S[i] + 2 * k3S[i]
                                + k4S[i])
            Iv[i] += h / 6.0 * (k1I[i] + 2 * k2I[i] + 2 * k3I[i]
                                + k4I[i])
            Sv[i] = min(max(Sv[i], 0.0), 1.0)
            Iv[i] = max(Iv[i], 0.0)
    return RichResult(payload={
        "estimate": list(Iv), "S": list(Sv), "I": list(Iv),
        "t": traj_t, "S_traj": traj_S, "I_traj": traj_I,
        "n_strains": n, "R0": basic_reproduction_numbers(b, nv, m),
        "n_variables": 2 * n,
        "n_variables_history_based": "2^%d = %d" % (n, 2 ** n)
        if n <= 30 else "2^%d" % n,
        "surviving": [i for i in range(n) if Iv[i] > 1e-8],
        "method": "status-based many-strain model, Gog & Grenfell "
                  "(2002), integrated by RK4",
    })


def linear_strain_space(n, width=2.0, floor=0.0):
    r"""Cross-immunity on a line, falling off with distance (Fig. 1).

    :math:`\sigma_{ij} = \exp(-(i-j)^2 / \text{width}^2)`, clipped
    below at ``floor``. The diagonal is 1 -- infection always immunises
    against itself.
    """
    if int(n) < 1:
        raise ValueError("hiatus: n must be at least 1")
    if float(width) <= 0.0:
        raise ValueError("hiatus: width must be positive")
    w = float(width)
    return [[max(float(floor),
                 math.exp(-((i - j) ** 2) / (w * w)))
             for j in range(int(n))] for i in range(int(n))]


def cheatsheet():
    return ("hiatus: many-strain dynamics in 2n variables, not 2^n. "
            "Status-based + reduced transmission + POLARIZED immunity "
            "(some hosts fully immune, not all partly) means one "
            "variable per host per strain. dI_i = b_i S_i I_i - "
            "(v_i + mu) I_i; dS_i = mu - sum_j b_j S_i sigma_ij I_j - "
            "mu S_i. sigma_ij = P(infection by j immunises against i). "
            "Off-diagonal sigma = 0 decouples the strains exactly; "
            "sigma = 1 everywhere gives competitive exclusion.")


# compact alias per ledger/NAMING.md
twostrainhiatus = simulate
