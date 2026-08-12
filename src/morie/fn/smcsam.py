r"""Sequential Monte Carlo samplers.

Del Moral, P., Doucet, A., & Jasra, A. (2006) "Sequential Monte Carlo
samplers", *Journal of the Royal Statistical Society: Series B* 68(3),
411-436.

The trick of the paper is to turn a *sequence of distributions on a fixed
space*, :math:`\pi_n(x) = \gamma_n(x)/Z_n`, into a sequential problem that
particle methods can attack. Sampling :math:`\pi_n` directly by importance
sampling has no sequential structure, so the authors build the artificial
joint target

.. math:: \tilde\pi_n(x_{1:n}) = \frac{\gamma_n(x_n)}{Z_n}
          \prod_{k=1}^{n-1} L_k(x_{k+1}, x_k)

with **backward** kernels :math:`L_k`, whose marginal in :math:`x_n` is
exactly :math:`\pi_n`. Importance sampling between
:math:`\eta_n(x_{1:n}) = \eta_1(x_1)\prod_k K_k(x_{k-1}, x_k)` and
:math:`\tilde\pi_n` then gives a weight that updates recursively
(equations 11-12):

.. math:: w_n(x_{1:n}) = w_{n-1}(x_{1:n-1})\,
          \tilde w_n(x_{n-1}, x_n), \qquad
          \tilde w_n(x_{n-1}, x_n) =
          \frac{\gamma_n(x_n) L_{n-1}(x_n, x_{n-1})}
               {\gamma_{n-1}(x_{n-1}) K_n(x_{n-1}, x_n)}.

**The case that makes it practical.** When :math:`K_n` is an MCMC kernel of
invariant distribution :math:`\pi_n`, the natural backward kernel
(equation 30) is the reversal
:math:`L_{n-1}(x_n, x_{n-1}) = \pi_n(x_{n-1})K_n(x_{n-1}, x_n)/\pi_n(x_n)`,
and the incremental weight collapses to something that does not involve the
kernel at all (equation 31):

.. math:: \tilde w_n(x_{n-1}, x_n) =
          \frac{\gamma_n(x_{n-1})}{\gamma_{n-1}(x_{n-1})}.

That is what this module uses by default: the weight is computed at the
particle's *previous* position, before the move, so any :math:`\pi_n`
-invariant kernel may be plugged in without changing the weighting.
``weight_rule="general"`` uses equation 12 instead, for a caller supplying
explicit forward and backward kernel densities.

**Degeneracy** is measured by the effective sample size,
:math:`\mathrm{ESS} = (\sum_i (W_n^{(i)})^2)^{-1}`, which "takes values
between 1 and N"; when it falls below a threshold, "say N/2", the particles
are resampled and reset to equal weights. Multinomial, stratified and
residual schemes are all implemented, the paper noting that the latter two
"reduce the variance of :math:`N_n^{(i)}` relatively to that of the
multinomial scheme".

**Normalising constants** come free: the incremental weights average to the
ratio :math:`Z_n/Z_{n-1}`, so their running product estimates
:math:`Z_n/Z_1`. That is the quantity the tempering sequence
:math:`\gamma_n = \pi^{\phi_n}` is usually run for, and the module returns
its logarithm.

Three sequences from section 2.3.1 are provided by
:func:`temperature_ladder`: geometric annealing
:math:`\pi_n \propto \pi^{\phi_n}` (also the optimisation route -- see
:mod:`morie.fn.smcopt`), data tempering, and a caller-supplied ladder.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["smcsam", "smc_sampler", "sequential_mc_sampler", "ess", "resample",
           "temperature_ladder", "random_walk_kernel"]


def ess(weights):
    r"""Effective sample size :math:`(\sum_i W_i^2)^{-1}` on normalised
    weights; between 1 and :math:`N`."""
    tot = sum(weights)
    if tot <= 0:
        raise ValueError("smcsam: weights must have positive total mass")
    w = [v / tot for v in weights]
    return 1.0 / sum(v * v for v in w)


def resample(weights, rng, scheme="systematic"):
    """Indices of the resampled particles.

    ``multinomial`` samples :math:`N` draws from the weighted distribution;
    ``stratified`` and ``systematic`` place one draw in each of :math:`N`
    strata, which reduces the variance of the offspring counts as the paper
    notes; ``residual`` takes the integer parts deterministically and
    multinomially samples the remainder.
    """
    tot = sum(weights)
    if tot <= 0:
        raise ValueError("smcsam: weights must have positive total mass")
    w = [v / tot for v in weights]
    n = len(w)
    cum = []
    run = 0.0
    for v in w:
        run += v
        cum.append(run)

    def pick(u):
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    if scheme == "multinomial":
        return [pick(rng.random()) for _ in range(n)]
    if scheme == "stratified":
        return [pick((k + rng.random()) / n) for k in range(n)]
    if scheme == "systematic":
        u0 = rng.random()
        return [pick((k + u0) / n) for k in range(n)]
    if scheme == "residual":
        out = []
        counts = [int(n * v) for v in w]
        for i, c in enumerate(counts):
            out.extend([i] * c)
        rem = n - len(out)
        if rem > 0:
            resid = [n * w[i] - counts[i] for i in range(n)]
            s = sum(resid)
            rcum, run2 = [], 0.0
            for v in resid:
                run2 += v / s
                rcum.append(run2)
            for _ in range(rem):
                u = rng.random()
                lo, hi = 0, n - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if rcum[mid] < u:
                        lo = mid + 1
                    else:
                        hi = mid
                out.append(lo)
        return out
    raise ValueError("smcsam: scheme must be multinomial, stratified, "
                     "systematic or residual")


def temperature_ladder(n_steps, kind="geometric", power=1.0):
    r"""A :math:`\phi_n` ladder from 0 to 1 for
    :math:`\gamma_n = \pi^{\phi_n}`.

    ``geometric`` spaces :math:`\phi` evenly, ``power`` raises the even
    spacing to ``power`` (``power > 1`` concentrates the steps near the
    end, where the target sharpens), and ``prior`` puts every step at 1
    except the first, which is the degenerate ladder worth comparing
    against.
    """
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("smcsam: need at least two steps")
    if kind == "geometric":
        return [t / float(n_steps - 1) for t in range(n_steps)]
    if kind == "power":
        if power <= 0:
            raise ValueError("smcsam: power must be positive")
        return [(t / float(n_steps - 1)) ** power for t in range(n_steps)]
    if kind == "prior":
        return [0.0] + [1.0] * (n_steps - 1)
    raise ValueError("smcsam: kind must be geometric, power or prior")


def random_walk_kernel(scale=1.0, n_moves=1):
    r"""A Metropolis random walk, returned as a :math:`\pi_n`-invariant
    kernel.

    The kernel signature is ``move(x, log_target, rng) -> (x_new,
    accepted)``; invariance is what equation 31 requires, and a symmetric
    proposal with the Metropolis rule provides it.
    """
    def move(x, log_target, rng):
        cur = list(x)
        lp = log_target(cur)
        acc = 0
        for _ in range(int(n_moves)):
            prop = [cur[k] + scale * rng.standard_normal()
                    for k in range(len(cur))]
            lq = log_target(prop)
            if math.log(max(rng.random(), 1e-300)) < lq - lp:
                cur, lp = prop, lq
                acc += 1
        return cur, acc / float(n_moves)
    return move


def smcsam(log_gamma, initial, n_particles=500, ladder=None, n_steps=20,
           kernel=None, ess_threshold=0.5, scheme="systematic", seed=0,
           weight_rule="mcmc", log_forward=None, log_backward=None):
    r"""Run an SMC sampler over :math:`\pi_n \propto \gamma_n`.

    Parameters
    ----------
    log_gamma : callable
        ``log_gamma(x, phi) -> float``, the log of the unnormalised
        :math:`\gamma_n` at temperature :math:`\phi_n`. For the usual
        annealing case this is
        ``log_prior(x) + phi * log_likelihood(x)``.
    initial : callable
        ``initial(rng) -> x``, a draw from :math:`\pi_1` (or from the
        prior, with ``ladder[0] = 0``).
    n_particles : int
        :math:`N`.
    ladder : sequence of float, optional
        The :math:`\phi_n`. Defaults to a geometric ladder of ``n_steps``.
    kernel : callable, optional
        ``move(x, log_target, rng) -> (x_new, accept_rate)``, invariant for
        the current :math:`\pi_n`. Defaults to a random walk.
    ess_threshold : float
        Resample when the ESS falls below this fraction of :math:`N`
        ("say N/2").
    scheme : str
        Resampling scheme; see :func:`resample`.
    weight_rule : {"mcmc", "general"}
        ``"mcmc"`` uses equation 31, valid when the kernel is
        :math:`\pi_n`-invariant. ``"general"`` uses equation 12 and needs
        ``log_forward`` and ``log_backward``.

    Returns
    -------
    RichResult
        ``estimate`` / ``mean`` is the weighted particle mean, ``particles``
        and ``weights`` the final approximation, ``log_norm_const`` the
        running estimate of :math:`\log(Z_n/Z_1)`, ``ess_trace``,
        ``resampled`` (which steps resampled), ``accept_trace``.

    Examples
    --------
    A Gaussian target reached by annealing from a broad prior::

        r = smcsam(lambda x, p: -0.5 * x[0] ** 2 * p
                                - 0.5 * (x[0] / 10.0) ** 2 * (1 - p),
                   lambda rng: [10.0 * rng.standard_normal()])
        r["mean"]

    References
    ----------
    Del Moral, Doucet & Jasra (2006) *JRSS-B* 68(3), 411-436: equations
    10-12, 30-31, the ESS criterion and section 3.1.1.
    """
    if weight_rule not in ("mcmc", "general"):
        raise ValueError("smcsam: weight_rule must be 'mcmc' or 'general'")
    if weight_rule == "general" and (log_forward is None or
                                     log_backward is None):
        raise ValueError("smcsam: weight_rule='general' needs log_forward "
                         "and log_backward densities (equation 12)")
    phis = list(ladder) if ladder is not None else temperature_ladder(
        n_steps)
    if len(phis) < 2:
        raise ValueError("smcsam: the ladder needs at least two steps")
    N = int(n_particles)
    if N < 2:
        raise ValueError("smcsam: need at least two particles")
    if not 0.0 < float(ess_threshold) <= 1.0:
        raise ValueError("smcsam: ess_threshold must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    move = kernel if kernel is not None else random_walk_kernel()

    X = [initial(rng) for _ in range(N)]
    logW = [0.0] * N
    log_norm = 0.0
    ess_trace, resampled, accept_trace = [], [], []

    for n in range(1, len(phis)):
        prev, cur = phis[n - 1], phis[n]
        # incremental weights BEFORE the move (equation 31)
        if weight_rule == "mcmc":
            inc = [log_gamma(X[i], cur) - log_gamma(X[i], prev)
                   for i in range(N)]
        else:
            inc = None
        if inc is not None:
            mx = max(inc)
            wprev = [math.exp(v) for v in
                     [lw - max(logW) for lw in logW]]
            tot_prev = sum(wprev)
            log_norm += mx + math.log(
                sum(wprev[i] * math.exp(inc[i] - mx) for i in range(N)) /
                tot_prev)
            logW = [logW[i] + inc[i] for i in range(N)]

        def target(x, _c=cur):
            return log_gamma(x, _c)

        moved = []
        acc = 0.0
        for i in range(N):
            xn, a = move(X[i], target, rng)
            if weight_rule == "general":
                num = log_gamma(xn, cur) + log_backward(xn, X[i], cur)
                den = log_gamma(X[i], prev) + log_forward(X[i], xn, cur)
                logW[i] += num - den
            moved.append(xn)
            acc += a
        X = moved
        accept_trace.append(acc / N)

        mx = max(logW)
        w = [math.exp(v - mx) for v in logW]
        e = ess(w)
        ess_trace.append(e)
        if e < ess_threshold * N:
            idx = resample(w, rng, scheme)
            X = [list(X[i]) for i in idx]
            logW = [0.0] * N
            resampled.append(n)

    mx = max(logW)
    w = [math.exp(v - mx) for v in logW]
    tot = sum(w)
    W = [v / tot for v in w]
    dim = len(X[0])
    mean = [sum(W[i] * X[i][k] for i in range(N)) for k in range(dim)]
    var = [sum(W[i] * (X[i][k] - mean[k]) ** 2 for i in range(N))
           for k in range(dim)]
    return RichResult(payload={
        "estimate": mean,
        "mean": mean,
        "variance": var,
        "particles": X,
        "weights": W,
        "log_norm_const": log_norm,
        "ess": ess(w),
        "ess_trace": ess_trace,
        "resampled": resampled,
        "accept_trace": accept_trace,
        "ladder": phis,
        "n_particles": N,
        "weight_rule": weight_rule,
        "method": "SMC sampler (Del Moral, Doucet & Jasra 2006)",
    })


def cheatsheet():
    return ("smcsam: SMC samplers (Del Moral, Doucet & Jasra 2006). A "
            "sequence pi_n on a FIXED space is made sequential by an "
            "artificial joint target built from BACKWARD kernels L_k, so "
            "the weights update recursively (eqs.11-12). With an MCMC "
            "kernel of invariant distribution pi_n, the natural L is its "
            "reversal (eq.30) and the incremental weight collapses to "
            "gamma_n(x_{n-1})/gamma_{n-1}(x_{n-1}) (eq.31) -- evaluated "
            "BEFORE the move, and free of the kernel. Degeneracy watched "
            "by ESS = 1/sum W^2, resample below N/2. The running product "
            "of incremental weights estimates Z_n/Z_1.")


# compact alias per ledger/NAMING.md
smc_sampler = smcsam

# name carried over from the generated stub this replaced
sequential_mc_sampler = smcsam
