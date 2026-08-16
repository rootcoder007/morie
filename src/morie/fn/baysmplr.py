"""Sampler dispatch: Metropolis, Gibbs, HMC and NUTS, and choosing among them.

The dispatcher is the point of this module, but a dispatcher that cannot
run what it recommends is a lookup table, so all four samplers are here
and the choice is made on properties of the problem rather than on
taste.

What actually distinguishes them:

  Metropolis-Hastings   needs only the log density. Its random walk
                        explores at a rate that degrades as the
                        dimension grows -- the optimal scaling result is
                        that the step must shrink like 1/sqrt(d) and the
                        acceptance rate settle near 0.234 -- so the cost
                        of an effectively independent draw grows like
                        d^2. Fine in three dimensions, hopeless in
                        three hundred.
  Gibbs                 needs conditional samplers, which usually means
                        conjugacy. When they exist it accepts
                        everything and has no tuning at all; when the
                        coordinates are strongly correlated it still
                        crawls, because every move is axis-aligned.
  HMC                   needs the GRADIENT. It follows Hamiltonian
                        dynamics, so a proposal can travel a long way
                        and still be accepted, and the cost per
                        independent draw grows like d^(5/4) rather than
                        d^2. The price is two numbers to tune: the step
                        size and the number of steps, and the second is
                        genuinely hard -- too few and it is a random
                        walk again, too many and it doubles back on
                        itself.
  NUTS                  removes the second of those. It doubles the
                        trajectory until it starts to turn back on
                        itself, then samples a point from what it built.
                        Same requirement as HMC -- a gradient -- and no
                        trajectory length to choose.

The dispatch rule, stated so it can be disagreed with:

    gradient and d >= 20        -> NUTS
    gradient and d <  20        -> HMC
    no gradient, conditionals   -> Gibbs
    no gradient, d <= 5         -> Metropolis-Hastings
    no gradient, d >  5         -> Metropolis-Hastings, with a warning
                                   that it will mix badly and that
                                   supplying a gradient is worth more
                                   than any amount of tuning

The threshold at 20 is a judgement, not a theorem: below it HMC's fixed
trajectory is easy enough to set and cheaper per draw than NUTS's
doubling, above it choosing the length by hand stops being reasonable.
It is exposed as `nuts_threshold` so it can be moved. Every result
carries `reason`, the sentence that explains the choice, because a
dispatcher that will not say why it chose is not usable in an argument.

Diagnostics are computed the same way for every sampler -- acceptance
rate, effective sample size by the initial-positive-sequence rule, and
the Geyer-truncated autocorrelation sum -- so the samplers can actually
be compared rather than each being scored on its own terms.

References
  Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H. and
    Teller, E. (1953) "Equation of state calculations by fast computing
    machines." Journal of Chemical Physics 21(6), 1087-1092.
  Hastings, W.K. (1970) "Monte Carlo sampling methods using Markov
    chains and their applications." Biometrika 57(1), 97-109.
  Geman, S. and Geman, D. (1984) "Stochastic relaxation, Gibbs
    distributions, and the Bayesian restoration of images." IEEE PAMI
    6(6), 721-741.
  Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D. (1987)
    "Hybrid Monte Carlo." Physics Letters B 195(2), 216-222.
  Neal, R.M. (2011) "MCMC using Hamiltonian dynamics." Handbook of
    Markov Chain Monte Carlo, chapter 5.
  Hoffman, M.D. and Gelman, A. (2014) "The No-U-Turn Sampler."
    Journal of Machine Learning Research 15, 1593-1623. Algorithms 3
    and 6: efficient NUTS and dual averaging.
  Roberts, G.O., Gelman, A. and Gilks, W.R. (1997) "Weak convergence
    and optimal scaling of random walk Metropolis algorithms." Annals of
    Applied Probability 7(1), 110-120. The 0.234 rate.
  Geyer, C.J. (1992) "Practical Markov chain Monte Carlo." Statistical
    Science 7(4), 473-483. The initial positive sequence estimator.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["sampler_dispatch", "baysmplr", "metropolis", "gibbs_normal",
           "hmc", "nuts", "effective_sample_size", "choose_sampler",
           "SAMPLERS", "cheatsheet"]

SAMPLERS = ("mh", "gibbs", "hmc", "nuts")


def choose_sampler(dim, has_grad, has_conditionals=False,
                   nuts_threshold=20):
    """The dispatch decision and the sentence that explains it."""
    if has_grad and dim >= nuts_threshold:
        return "nuts", ("gradient available and dimension %d at or above "
                        "the threshold %d, where choosing a trajectory "
                        "length by hand stops being reasonable"
                        % (dim, nuts_threshold))
    if has_grad:
        return "hmc", ("gradient available and dimension %d below the "
                       "threshold %d, where a fixed trajectory is easy "
                       "to set and cheaper per draw than NUTS doubling"
                       % (dim, nuts_threshold))
    if has_conditionals:
        return "gibbs", ("no gradient, but exact conditionals are "
                         "available, so every move is accepted and "
                         "nothing needs tuning")
    if dim <= 5:
        return "mh", ("no gradient and dimension %d is small, so a "
                      "random walk is adequate" % dim)
    return "mh", ("no gradient and dimension %d is large; a random walk "
                  "will mix badly, and supplying a gradient would be "
                  "worth more than any amount of tuning" % dim)


def metropolis(log_p, x0, n_iter, rng, scale=None, adapt=False,
               target_accept=0.234):
    """Random-walk Metropolis with an isotropic normal proposal.

    The default scale is 2.38 / sqrt(d), the optimal-scaling value for a
    product target; `adapt` then tunes it towards `target_accept` by
    Robbins-Monro on the log scale during the first half of the run,
    after which it is frozen so the chain that is kept is homogeneous.
    """
    d = len(x0)
    if scale is None:
        scale = 2.38 / math.sqrt(d)
    x = list(x0)
    lp = log_p(x)
    draws = []
    acc = 0
    half = int(n_iter) // 2
    for it in range(int(n_iter)):
        prop = [x[j] + scale * float(rng.normal()) for j in range(d)]
        lq = log_p(prop)
        a = lq - lp
        if a >= 0.0 or math.log(float(rng.uniform())) < a:
            x, lp = prop, lq
            acc += 1
            ar = 1.0
        else:
            ar = 0.0
        if adapt and it < half:
            scale = math.exp(math.log(scale)
                             + (ar - target_accept) / math.sqrt(it + 1.0))
        draws.append(list(x))
    return draws, acc / float(n_iter), {"scale": scale}


def gibbs_normal(mean, cov_inv, x0, n_iter, rng):
    """Gibbs on a multivariate normal, using its exact conditionals.

    The conditional of coordinate j given the rest is normal with
    precision Q_jj and mean mu_j - (1/Q_jj) sum_{k != j} Q_jk (x_k -
    mu_k). Written from the precision matrix because that is the form
    in which the conditionals are trivial -- inverting a covariance to
    get them and then inverting back is the usual way to make this
    slower and less accurate than it needs to be.
    """
    d = len(x0)
    x = list(x0)
    draws = []
    for _ in range(int(n_iter)):
        for j in range(d):
            s = _w.csum(cov_inv[j][k] * (x[k] - mean[k])
                        for k in range(d) if k != j)
            mj = mean[j] - s / cov_inv[j][j]
            sj = math.sqrt(1.0 / cov_inv[j][j])
            x[j] = mj + sj * float(rng.normal())
        draws.append(list(x))
    # Gibbs accepts everything by construction; reporting 1.0 is a
    # statement about the algorithm, not a measurement.
    return draws, 1.0, {}


def _leapfrog(grad, q, p, eps, steps):
    q = list(q)
    p = list(p)
    g = grad(q)
    d = len(q)
    for _ in range(int(steps)):
        p = [p[j] + 0.5 * eps * g[j] for j in range(d)]
        q = [q[j] + eps * p[j] for j in range(d)]
        g = grad(q)
        p = [p[j] + 0.5 * eps * g[j] for j in range(d)]
    return q, p, g


def hmc(log_p, grad, x0, n_iter, rng, eps=0.1, steps=10):
    """Hamiltonian Monte Carlo with a fixed trajectory.

    The momentum is refreshed from a standard normal each iteration and
    the leapfrog integrator is used because it is symplectic and
    reversible: the acceptance step corrects only its energy error, and
    a non-reversible integrator would leave a bias no acceptance rule
    can remove.
    """
    d = len(x0)
    x = list(x0)
    lp = log_p(x)
    draws = []
    acc = 0
    for _ in range(int(n_iter)):
        p0 = [float(rng.normal()) for _ in range(d)]
        k0 = 0.5 * _w.csum(v * v for v in p0)
        q, p, _g = _leapfrog(grad, x, p0, eps, steps)
        lq = log_p(q)
        k1 = 0.5 * _w.csum(v * v for v in p)
        a = (lq - k1) - (lp - k0)
        if a >= 0.0 or math.log(float(rng.uniform())) < a:
            x, lp = q, lq
            acc += 1
        draws.append(list(x))
    return draws, acc / float(n_iter), {"eps": eps, "steps": steps}


def _build_tree(log_p, grad, q, p, u, v, j, eps, rng, h0, dmax=1000.0):
    """Hoffman and Gelman's recursive doubling, Algorithm 3."""
    if j == 0:
        qp, pp, _ = _leapfrog(grad, q, p, v * eps, 1)
        h = log_p(qp) - 0.5 * _w.csum(t * t for t in pp)
        n = 1 if u <= math.exp(h) else 0
        s = 1 if h > math.log(u) - dmax else 0
        return (qp, pp, qp, pp, qp, n, s,
                min(1.0, math.exp(h - h0)), 1)
    (qm, pm, qpl, ppl, qpr, n1, s1, a1, na1) = _build_tree(
        log_p, grad, q, p, u, v, j - 1, eps, rng, h0)
    if s1 == 1:
        if v == -1:
            (qm, pm, _x, _y, q2, n2, s2, a2, na2) = _build_tree(
                log_p, grad, qm, pm, u, v, j - 1, eps, rng, h0)
        else:
            (_x, _y, qpl, ppl, q2, n2, s2, a2, na2) = _build_tree(
                log_p, grad, qpl, ppl, u, v, j - 1, eps, rng, h0)
        if n1 + n2 > 0 and float(rng.uniform()) < n2 / float(n1 + n2):
            qpr = q2
        a1 += a2
        na1 += na2
        # The no-U-turn condition: stop when the trajectory's two ends
        # are moving towards each other rather than apart.
        dq = [qpl[k] - qm[k] for k in range(len(qm))]
        s1 = s2 * (1 if _w.dot(dq, pm) >= 0.0 else 0) \
            * (1 if _w.dot(dq, ppl) >= 0.0 else 0)
        n1 += n2
    return (qm, pm, qpl, ppl, qpr, n1, s1, a1, na1)


def nuts(log_p, grad, x0, n_iter, rng, eps=0.25, max_depth=8,
         dual_average=False, target_accept=0.8, warmup=None):
    """The No-U-Turn Sampler, with optional dual-averaging warm-up.

    The trajectory doubles until the two ends start approaching each
    other, which is what removes the trajectory-length parameter. With
    `dual_average` the step size is also tuned during warm-up by
    Hoffman and Gelman's Algorithm 6 and then frozen, so the retained
    chain has a single fixed step size rather than a moving one.
    """
    d = len(x0)
    x = list(x0)
    draws = []
    depths = []
    accs = []
    if warmup is None:
        warmup = int(n_iter) // 2
    mu = math.log(10.0 * eps)
    log_eps_bar = 0.0
    hbar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    for it in range(int(n_iter)):
        p0 = [float(rng.normal()) for _ in range(d)]
        h0 = log_p(x) - 0.5 * _w.csum(v * v for v in p0)
        u = float(rng.uniform()) * math.exp(h0)
        if u <= 0.0:
            u = 1e-300
        qm = list(x)
        qp = list(x)
        pm = list(p0)
        pp = list(p0)
        xnew = list(x)
        n = 1
        s = 1
        j = 0
        a_sum = 0.0
        na = 1
        while s == 1 and j < int(max_depth):
            v = -1 if float(rng.uniform()) < 0.5 else 1
            if v == -1:
                (qm, pm, _a, _b, q2, n2, s2, a2, na2) = _build_tree(
                    log_p, grad, qm, pm, u, v, j, eps, rng, h0)
            else:
                (_a, _b, qp, pp, q2, n2, s2, a2, na2) = _build_tree(
                    log_p, grad, qp, pp, u, v, j, eps, rng, h0)
            if s2 == 1 and n > 0 and float(rng.uniform()) < n2 / float(n):
                xnew = q2
            a_sum += a2
            na += na2
            n += n2
            dq = [qp[k] - qm[k] for k in range(d)]
            s = s2 * (1 if _w.dot(dq, pm) >= 0.0 else 0) \
                * (1 if _w.dot(dq, pp) >= 0.0 else 0)
            j += 1
        x = xnew
        depths.append(j)
        accs.append(a_sum / na if na else 0.0)
        if dual_average and it < warmup:
            m = it + 1.0
            hbar = ((1.0 - 1.0 / (m + t0)) * hbar
                    + (target_accept - a_sum / na) / (m + t0))
            log_eps = mu - math.sqrt(m) / gamma * hbar
            w = math.pow(m, -kappa)
            log_eps_bar = w * log_eps + (1.0 - w) * log_eps_bar
            eps = math.exp(log_eps)
        elif dual_average and it == warmup:
            eps = math.exp(log_eps_bar)
        draws.append(list(x))
    return draws, _w.csum(accs) / len(accs), {"eps": eps,
                                              "mean_depth":
                                              _w.csum(float(v) for v in
                                                      depths)
                                              / len(depths)}


def effective_sample_size(chain, max_lag=200):
    """Geyer's initial positive sequence estimator, per coordinate.

    Sums the autocorrelations in adjacent PAIRS and stops when a pair
    turns negative. The pairing is not decoration: for a reversible
    chain the sum of two adjacent autocorrelations is positive, so
    truncating on a single negative lag stops too early on a noisy tail
    and inflates the answer.

    Lags beyond `max_lag` are not computed. The estimator is meant to
    stop long before that -- if it has not, the chain has an
    autocorrelation time comparable to its own length and the number it
    would return is not trustworthy anyway.
    """
    n = len(chain)
    d = len(chain[0])
    out = []
    for c in range(d):
        v = [row[c] for row in chain]
        mu = _w.csum(v) / n
        dev = [t - mu for t in v]
        var = _w.csum(t * t for t in dev) / n
        if var <= 0.0:
            out.append(float(n))
            continue
        rho = []
        top = n - 1 if n - 1 < int(max_lag) else int(max_lag)
        for lag in range(1, top + 1):
            rho.append(_w.csum(dev[i] * dev[i + lag]
                               for i in range(n - lag)) / (n * var))
        total = 0.0
        k = 0
        while k + 1 < len(rho):
            pair = rho[k] + rho[k + 1]
            if pair <= 0.0:
                break
            total += pair
            k += 2
        out.append(n / (1.0 + 2.0 * total) if 1.0 + 2.0 * total > 0.0
                   else float(n))
    return out


def sampler_dispatch(log_p, grad_p=None, x0=None, n_iter=500, burn=None,
                     seed=1, sampler=None, cov_inv=None, mean=None,
                     eps=0.1, steps=10, scale=None, adapt=False,
                     nuts_threshold=20, dual_average=False,
                     max_depth=8):
    """Pick a sampler for this problem and run it.

    Parameters
    ----------
    log_p : callable
        Log density, up to a constant, of a list of floats.
    grad_p : callable or None
        Its gradient. Supplying it is what makes HMC and NUTS available,
        and the dispatcher will take them when it can.
    x0 : sequence
        Starting point. Its length is the dimension.
    n_iter, burn : int
        Iterations and burn-in. Burn-in defaults to half.
    seed : int
        Seed for the generator shared with the R arm.
    sampler : str or None
        Force a sampler instead of dispatching. The reason field then
        records that the choice was overridden.
    cov_inv, mean : sequences or None
        A normal target's precision matrix and mean, which is what makes
        the Gibbs route available.
    eps, steps : float, int
        HMC step size and trajectory length.
    scale : float or None
        Metropolis proposal scale.
    adapt : bool
        Tune the Metropolis scale during the first half.
    nuts_threshold : int
        Dimension at or above which NUTS is preferred to HMC.
    dual_average : bool
        Tune the NUTS step size during warm-up.
    max_depth : int
        NUTS doubling limit.

    Returns
    -------
    RichResult
        The chosen sampler and the reason, the posterior mean and
        standard deviation per coordinate, the acceptance rate, the
        effective sample size, and the retained draws.

    References
    ----------
    Hoffman and Gelman (2014) JMLR 15, 1593-1623; Neal (2011) Handbook
    of MCMC ch. 5; Roberts, Gelman and Gilks (1997) Ann. Appl. Probab.
    7(1), 110-120; Geyer (1992) Statist. Sci. 7(4), 473-483.
    """
    if x0 is None:
        raise ValueError("x0 is required; its length is the dimension")
    x0 = [float(v) for v in x0]
    d = len(x0)
    if d < 1:
        raise ValueError("x0 must be non-empty")
    if burn is None:
        burn = int(n_iter) // 2
    if burn >= int(n_iter):
        raise ValueError("burn-in consumes every iteration")
    has_cond = cov_inv is not None and mean is not None
    if sampler is None:
        sampler, reason = choose_sampler(d, grad_p is not None, has_cond,
                                         nuts_threshold)
    else:
        if sampler not in SAMPLERS:
            raise ValueError("sampler must be one of %r" % (SAMPLERS,))
        reason = "forced by the caller, dispatch not consulted"
    if sampler in ("hmc", "nuts") and grad_p is None:
        raise ValueError("%s needs a gradient" % sampler)
    if sampler == "gibbs" and not has_cond:
        raise ValueError("gibbs needs mean and cov_inv")

    rng = _core._SplitMix64(seed)
    if sampler == "mh":
        draws, acc, info = metropolis(log_p, x0, n_iter, rng, scale,
                                      adapt)
    elif sampler == "gibbs":
        draws, acc, info = gibbs_normal(mean, cov_inv, x0, n_iter, rng)
    elif sampler == "hmc":
        draws, acc, info = hmc(log_p, grad_p, x0, n_iter, rng, eps,
                               steps)
    else:
        draws, acc, info = nuts(log_p, grad_p, x0, n_iter, rng, eps,
                                max_depth, dual_average)

    kept = draws[int(burn):]
    m = len(kept)
    means = [_w.csum(row[c] for row in kept) / m for c in range(d)]
    sds = []
    for c in range(d):
        if m > 1:
            sds.append(math.sqrt(_w.csum((row[c] - means[c])
                                         * (row[c] - means[c])
                                         for row in kept) / (m - 1)))
        else:
            sds.append(0.0)
    ess = effective_sample_size(kept)

    return RichResult(payload={
        "sampler": sampler,
        "reason": reason,
        "mean": means,
        "sd": sds,
        "ess": ess,
        "min_ess": min(ess),
        "ess_per_draw": min(ess) / m,
        "accept_rate": acc,
        "draws": kept,
        "kept": m,
        "dim": d,
        "n_iter": int(n_iter),
        "burn": int(burn),
        "info": info,
        "seed": int(seed),
        "estimate": means[0],
        "method": "MCMC sampler dispatch",
    })


baysmplr = sampler_dispatch


def cheatsheet():
    return ("baysmplr: MCMC sampler dispatch and the samplers themselves. "
            + ", ".join(SAMPLERS))
