# morie.fn -- function file (rootcoder007/morie)
r"""The No-U-Turn Sampler, with dual-averaging step-size adaptation.

Hoffman, M. D., & Gelman, A. (2014) "The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo", *Journal of Machine
Learning Research* 15, 1593-1623.

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H.,
& Teller, E. (1953) "Equation of State Calculations by Fast Computing
Machines", *Journal of Chemical Physics* 21(6), 1087-1092.
doi:10.1063/1.1699114 -- the accept-reject rule the energy error is
corrected with below. NUTS replaces the fixed-length trajectory and
the hand-tuned step size, not this.

Hamiltonian Monte Carlo turns sampling into physics: augment the target
:math:`p(\theta)` with a momentum :math:`r \sim N(0, I)`, giving the
joint :math:`p(\theta, r) \propto \exp\{\mathcal{L}(\theta) -
\tfrac{1}{2} r \cdot r\}`, and simulate Hamiltonian dynamics with the
leapfrog integrator. Because the integrator is reversible and
volume-preserving, a Metropolis correction of the energy error is all
that is needed, and long trajectories move far without the random walk
of a Metropolis sampler.

That leaves two dials the user should not have to turn.

**How long to run the trajectory** -- the paper's answer, and its name.
Keep doubling the trajectory (forwards or backwards, chosen at random,
so the construction stays reversible) until it starts to double back on
itself. The U-turn is detected by the sign of

.. math::

   (\theta^{+} - \theta^{-}) \cdot r^{-} \quad\text{and}\quad
   (\theta^{+} - \theta^{-}) \cdot r^{+},

either going negative meaning the ends have begun to approach. States
are then sampled from the trajectory with the progressive scheme of
Algorithm 3, which keeps the slice-sampling correctness of Algorithm 2
without storing every point. A state also dies if the simulation error
blows past :math:`\Delta_{max}`, which the paper recommends setting to
1000.

**How big a step to take** -- dual averaging (Nesterov), Equation 6:

.. math::

   x_{t+1} \leftarrow \mu - \frac{\sqrt{t}}{\gamma}\,
   \frac{1}{t + t_0}\sum_{i=1}^{t} H_i, \qquad
   \bar{x}_{t+1} \leftarrow \eta_t x_{t+1} + (1 - \eta_t)\bar{x}_t,

with :math:`\eta_t = t^{-\kappa}`, driving the statistic
:math:`H_t = \delta - \alpha_t` to zero -- that is, driving the average
Metropolis acceptance to the target :math:`\delta`. The paper's values
are used: :math:`\gamma = 0.05`, :math:`t_0 = 10`,
:math:`\kappa = 0.75`, :math:`\delta = 0.65` (the optimum for HMC under
fairly strong assumptions, Beskos et al.), and :math:`\mu = \log(10
\epsilon_1)` with :math:`\epsilon_1` from Algorithm 4, which doubles or
halves the step size until the acceptance probability crosses one half.
After warmup the averaged :math:`\bar{\epsilon}` is used and adaptation
stops, so the sampled chain is a proper Markov chain.

Both samplers are here (``sampler="nuts"`` or ``"hmc"``); HMC is the
Algorithm 1 baseline the paper measures against and needs its own
``n_steps``.

``logp`` is called as ``logp(theta)`` and must return the log density up
to a constant. Supply ``grad`` for the gradient, or leave it out and a
central difference is used -- exact enough for an anchor, wasteful for
real work.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "bayhmc",
    "hmc_nuts",
    "leapfrog",
    "find_reasonable_epsilon",
    "dual_averaging_update",
    "build_tree",
    "no_u_turn",
    "DELTA_MAX",
]

#: The paper's recommended cap on the simulation error.
DELTA_MAX = 1000.0

_SAMPLERS = ("nuts", "hmc")


def _numeric_grad(logp, theta, h=1e-5):
    out = []
    for i in range(len(theta)):
        up = list(theta)
        dn = list(theta)
        up[i] += h
        dn[i] -= h
        out.append((logp(up) - logp(dn)) / (2 * h))
    return out


def leapfrog(theta, r, eps, grad):
    r"""One leapfrog step: half-kick, drift, half-kick.

    The integrator is reversible and volume preserving, which is what
    lets a single Metropolis test on the energy correct it exactly.
    """
    g = grad(theta)
    r_half = [r[i] + 0.5 * eps * g[i] for i in range(len(r))]
    new_theta = [theta[i] + eps * r_half[i] for i in range(len(theta))]
    g2 = grad(new_theta)
    new_r = [r_half[i] + 0.5 * eps * g2[i] for i in range(len(r))]
    return new_theta, new_r


def _joint(logp, theta, r):
    return logp(theta) - 0.5 * sum(v * v for v in r)


def find_reasonable_epsilon(theta, logp, grad, rnd, eps=1.0,
                            max_doublings=100):
    """Algorithm 4: double or halve until the acceptance crosses 1/2."""
    r = [rnd() for _ in range(len(theta))]
    t2, r2 = leapfrog(theta, r, eps, grad)
    log_ratio = _joint(logp, t2, r2) - _joint(logp, theta, r)
    a = 1.0 if log_ratio > math.log(0.5) else -1.0
    for _ in range(int(max_doublings)):
        if a * log_ratio <= a * math.log(0.5):
            break
        eps = eps * (2.0 ** a)
        t2, r2 = leapfrog(theta, r, eps, grad)
        log_ratio = _joint(logp, t2, r2) - _joint(logp, theta, r)
    return eps


def dual_averaging_update(t, h_bar, log_eps_bar, h_new, mu, gamma=0.05,
                          t0=10.0, kappa=0.75):
    r"""Equation 6, one iteration.

    ``h_bar`` carries :math:`\frac{1}{t + t_0}\sum_i H_i` between calls.
    Returns ``(eps, h_bar, log_eps_bar)``.
    """
    if t < 1:
        raise ValueError("bayhmc: the dual averaging step must start at 1")
    if gamma <= 0 or t0 < 0 or not 0.5 < kappa <= 1.0:
        raise ValueError("bayhmc: gamma must be positive, t0 non-negative "
                         "and kappa in (0.5, 1]")
    eta = 1.0 / (t + t0)
    h_bar = (1.0 - eta) * h_bar + eta * h_new
    log_eps = mu - math.sqrt(t) / gamma * h_bar
    w = t ** (-kappa)
    log_eps_bar = w * log_eps + (1.0 - w) * log_eps_bar
    return math.exp(log_eps), h_bar, log_eps_bar


def no_u_turn(theta_minus, theta_plus, r_minus, r_plus):
    r"""The stopping rule: has the trajectory begun to double back?

    False once :math:`(\theta^{+} - \theta^{-}) \cdot r` goes negative at
    either end.
    """
    d = [theta_plus[i] - theta_minus[i] for i in range(len(theta_plus))]
    return (sum(d[i] * r_minus[i] for i in range(len(d))) >= 0 and
            sum(d[i] * r_plus[i] for i in range(len(d))) >= 0)


def build_tree(theta, r, logu, v, j, eps, logp, grad, rnd, joint0):
    """The recursion of Algorithm 3, returning the paper's seven values."""
    if j == 0:
        t2, r2 = leapfrog(theta, r, v * eps, grad)
        jj = _joint(logp, t2, r2)
        n = 1 if logu <= jj else 0
        s = 1 if jj > logu - DELTA_MAX else 0
        return (t2, r2, t2, r2, t2, n, s,
                min(1.0, math.exp(min(jj - joint0, 700.0))), 1)
    tm, rm, tp, rp, t_p, n_p, s_p, a_p, na_p = build_tree(
        theta, r, logu, v, j - 1, eps, logp, grad, rnd, joint0)
    if s_p == 1:
        if v == -1:
            tm, rm, _, _, t2, n2, s2, a2, na2 = build_tree(
                tm, rm, logu, v, j - 1, eps, logp, grad, rnd, joint0)
        else:
            _, _, tp, rp, t2, n2, s2, a2, na2 = build_tree(
                tp, rp, logu, v, j - 1, eps, logp, grad, rnd, joint0)
        if n_p + n2 > 0 and rnd() < n2 / float(n_p + n2):
            t_p = t2
        a_p += a2
        na_p += na2
        s_p = s2 if no_u_turn(tm, tp, rm, rp) else 0
        n_p += n2
    return tm, rm, tp, rp, t_p, n_p, s_p, a_p, na_p


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def uni():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    def normal():
        u1 = max(uni(), 1e-12)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi *
                                                         uni())
    return uni, normal


def bayhmc(logp, theta0, n_iter=1000, warmup=None, grad=None,
           sampler="nuts", delta=0.65, eps=None, n_steps=10,
           max_depth=10, gamma=0.05, t0=10.0, kappa=0.75, seed=0):
    """Sample ``logp`` by NUTS or HMC (Hoffman & Gelman 2014)."""
    if sampler not in _SAMPLERS:
        raise ValueError("bayhmc: sampler must be one of %s" % (_SAMPLERS,))
    theta = [float(v) for v in theta0]
    if not theta:
        raise ValueError("bayhmc: theta0 is empty")
    if n_iter < 1:
        raise ValueError("bayhmc: n_iter must be at least 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("bayhmc: delta must be in (0, 1)")
    if max_depth < 1 or n_steps < 1:
        raise ValueError("bayhmc: max_depth and n_steps must be positive")
    if eps is not None and eps <= 0:
        raise ValueError("bayhmc: eps must be positive")
    warm = n_iter // 2 if warmup is None else int(warmup)
    if not 0 <= warm <= n_iter:
        raise ValueError("bayhmc: warmup must be between 0 and n_iter")
    g = grad if grad is not None else (lambda t: _numeric_grad(logp, t))
    uni, normal = _rng(seed)

    if eps is None:
        eps = find_reasonable_epsilon(theta, logp, g, normal)
    mu = math.log(10.0 * eps)
    h_bar, log_eps_bar = 0.0, 0.0
    draws, accepts, depths, eps_trace = [], [], [], []

    for m in range(1, int(n_iter) + 1):
        r0 = [normal() for _ in range(len(theta))]
        joint0 = _joint(logp, theta, r0)
        if sampler == "hmc":
            t2, r2 = theta, r0
            for _ in range(int(n_steps)):
                t2, r2 = leapfrog(t2, r2, eps, g)
            alpha = min(1.0, math.exp(min(_joint(logp, t2, r2) - joint0,
                                          700.0)))
            if uni() < alpha:
                theta = t2
            depth = int(n_steps)
        else:
            logu = joint0 + math.log(max(uni(), 1e-300))
            tm = tp = theta
            rm = rp = r0
            j, n, s = 0, 1, 1
            alpha, n_alpha = 0.0, 1
            while s == 1 and j < int(max_depth):
                v = 1 if uni() < 0.5 else -1
                if v == -1:
                    tm, rm, _, _, t2, n2, s2, a, na = build_tree(
                        tm, rm, logu, v, j, eps, logp, g, uni, joint0)
                else:
                    _, _, tp, rp, t2, n2, s2, a, na = build_tree(
                        tp, rp, logu, v, j, eps, logp, g, uni, joint0)
                if s2 == 1 and n > 0 and uni() < min(1.0, n2 / float(n)):
                    theta = t2
                n += n2
                alpha, n_alpha = a, na
                s = s2 if no_u_turn(tm, tp, rm, rp) else 0
                j += 1
            depth = j
            alpha = alpha / float(max(n_alpha, 1))
        accepts.append(alpha)
        depths.append(depth)
        if m <= warm:
            eps, h_bar, log_eps_bar = dual_averaging_update(
                m, h_bar, log_eps_bar, delta - alpha, mu, gamma, t0,
                kappa)
        elif m == warm + 1:
            eps = math.exp(log_eps_bar) if warm > 0 else eps
        eps_trace.append(eps)
        if m > warm:
            draws.append(list(theta))
    if not draws:
        draws = [list(theta)]
    d = len(theta)
    n = float(len(draws))
    mean = [sum(x[i] for x in draws) / n for i in range(d)]
    var = [sum((x[i] - mean[i]) ** 2 for x in draws) / max(n - 1.0, 1.0)
           for i in range(d)]
    post = accepts[warm:] or accepts
    return RichResult(payload={
        "estimate": mean,
        "samples": draws,
        "mean": mean,
        "variance": var,
        "sd": [math.sqrt(v) for v in var],
        "acceptance": sum(post) / float(len(post)),
        "eps": eps,
        "eps_trace": eps_trace,
        "depths": depths,
        "n_samples": len(draws),
        "warmup": warm,
        "sampler": sampler,
        "delta": float(delta),
        "method": ("NUTS (Hoffman & Gelman 2014) with dual-averaging "
                   "step size" if sampler == "nuts" else
                   "Hamiltonian Monte Carlo (Algorithm 1)"),
        "note": ("adaptation runs during warmup only and the averaged "
                 "epsilon is used afterwards, so the sampled chain is a "
                 "proper Markov chain; gamma=0.05, t0=10, kappa=0.75 and "
                 "delta=0.65 are the paper's values, Delta_max=1000"),
    })


hmc_nuts = bayhmc


def cheatsheet():
    return ("bayhmc: NUTS (Hoffman & Gelman 2014). Hamiltonian dynamics "
            "by leapfrog, trajectory doubled at random forwards or "
            "backwards until (theta+ - theta-).r goes negative at either "
            "end -- the U-turn -- with progressive sampling from the "
            "trajectory. The step size is tuned by Nesterov dual "
            "averaging (eq.6) to hit an average acceptance of delta, "
            "during warmup only. sampler='hmc' gives the fixed-length "
            "Algorithm 1 baseline.")

# public names resolved by fn/_lazy_map.json
hmc_dual_avg = bayhmc
hmcdualavg = bayhmc
