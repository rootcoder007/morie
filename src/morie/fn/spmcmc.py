# morie.fn -- function file (rootcoder007/morie)
"""MCMC samplers for spatial posteriors on the Philox stream: the Poisson log-linear conditional
autoregressive (CAR) disease-mapping target, random-walk Metropolis-Hastings (joint or
single-site), Hamiltonian Monte Carlo, the No-U-Turn sampler and parallel tempering."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform

__all__ = ["car_poisson_target", "mh_spatial", "hmc_spatial", "nuts_spatial", "tempered_spatial"]


def car_poisson_target(y, expected, A, *, tau: float = 1.0, rho: float = 0.9, prior_sd: float = 10.0):
    r"""Log posterior and gradient of the Poisson log-linear proper-CAR disease-mapping model.

    ``y_i ~ Poisson(E_i exp(beta_0 + phi_i))``, ``phi ~ N(0, (tau (D - rho A))^(-1))``
    (Cressie 1993; Besag, York and Mollie 1991 with a proper CAR),
    ``beta_0 ~ N(0, prior_sd^2)``; the parameter vector is
    ``(beta_0, phi_1, ..., phi_n)``. Returns ``(logp, grad)`` callables.

    References
    ----------
    Banerjee, S., Carlin, B. P. and Gelfand, A. E. (2014). Hierarchical
    Modeling and Analysis for Spatial Data, 2nd ed., ch. 6. Lawson, A. B.
    (2018). Bayesian Disease Mapping, 3rd ed.

    Examples
    --------
    >>> lp, gr = car_poisson_target([1, 2], [1.0, 1.0], [[0, 1], [1, 0]])
    >>> round(lp([0.0, 0.0, 0.0]), 12)
    -2.0
    """
    ys = [float(v) for v in y]
    E = [float(v) for v in expected]
    n = len(ys)
    d = [ssum(float(v) for v in row) for row in A]
    Q = [[tau * ((d[i] if i == j else 0.0) - rho * float(A[i][j])) for j in range(n)] for i in range(n)]

    def logp(x):
        b, phi = x[0], x[1:]
        eta = [b + v for v in phi]
        ll = ssum(ys[i] * eta[i] - E[i] * math.exp(eta[i]) for i in range(n))
        qf = ssum(phi[i] * ssum(Q[i][j] * phi[j] for j in range(n)) for i in range(n))
        return ll - 0.5 * qf - 0.5 * b * b / prior_sd**2

    def grad(x):
        b, phi = x[0], x[1:]
        r = [ys[i] - E[i] * math.exp(b + phi[i]) for i in range(n)]
        gphi = [r[i] - ssum(Q[i][j] * phi[j] for j in range(n)) for i in range(n)]
        return [ssum(r) - b / prior_sd**2] + gphi

    return logp, grad


def mh_spatial(logp, x0, n_iter: int, *, step: float = 0.1, single_site: bool = False, seed: int = 0) -> RichResult:
    r"""Random-walk Metropolis-Hastings (joint Gaussian proposal, or single-site sweeps).

    Joint: ``x' = x + step z``, accepted with probability
    ``min(1, exp(logp(x') - logp(x)))``; iteration ``t`` draws ``z`` from
    Philox stream ``2t`` and the acceptance uniform(s) from stream ``2t + 1``.
    With ``single_site`` each coordinate is updated in turn (Metropolis within
    Gibbs), using entry ``j`` of both streams for coordinate ``j``.

    References
    ----------
    Metropolis, N. et al. (1953). J. Chem. Phys. 21, 1087-1092. Hastings, W.
    K. (1970). Biometrika 57, 97-109.

    Examples
    --------
    >>> r = mh_spatial(lambda x: -0.5 * x[0] ** 2, [0.0], 50, step=1.0, seed=1)
    >>> len(r.samples), 0.0 < r.acceptance <= 1.0
    (50, True)
    """
    x = [float(v) for v in x0]
    d = len(x)
    lp = logp(x)
    out, acc, tot = [], 0, 0
    for t in range(n_iter):
        z = random_normal(d, seed=seed, stream=2 * t)
        u = random_uniform(d if single_site else 1, seed=seed, stream=2 * t + 1)
        if single_site:
            for j in range(d):
                prop = list(x)
                prop[j] += step * float(z[j])
                lq = logp(prop)
                tot += 1
                if math.log(float(u[j])) < lq - lp:
                    x, lp = prop, lq
                    acc += 1
        else:
            prop = [a + step * float(b) for a, b in zip(x, z)]
            lq = logp(prop)
            tot += 1
            if math.log(float(u[0])) < lq - lp:
                x, lp = prop, lq
                acc += 1
        out.append(list(x))
    return RichResult(payload={"samples": out, "acceptance": acc / tot, "logp": lp})


def _leapfrog(grad, x, p, eps):
    g = grad(x)
    p = [a + 0.5 * eps * b for a, b in zip(p, g)]
    x = [a + eps * b for a, b in zip(x, p)]
    g = grad(x)
    p = [a + 0.5 * eps * b for a, b in zip(p, g)]
    return x, p


def hmc_spatial(logp, grad, x0, n_iter: int, *, eps: float = 0.05, n_leapfrog: int = 20, seed: int = 0) -> RichResult:
    r"""Hamiltonian Monte Carlo with an identity mass matrix (Duane et al. 1987; Neal 2011).

    Momentum ``p ~ N(0, I)`` (Philox stream ``2t``), ``n_leapfrog`` leapfrog
    steps of size ``eps``, acceptance with probability
    ``min(1, exp(H(x, p) - H(x', p')))``, ``H = -logp + |p|^2 / 2`` (uniform
    from stream ``2t + 1``).

    References
    ----------
    Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In Handbook of
    Markov Chain Monte Carlo, ch. 5. Duane, S. et al. (1987). Phys. Lett. B 195, 216-222.

    Examples
    --------
    >>> r = hmc_spatial(lambda x: -0.5 * x[0] ** 2, lambda x: [-x[0]], [0.5], 20, eps=0.2, n_leapfrog=5)
    >>> len(r.samples)
    20
    """
    x = [float(v) for v in x0]
    d = len(x)
    lp = logp(x)
    out, acc = [], 0
    for t in range(n_iter):
        p0 = [float(v) for v in random_normal(d, seed=seed, stream=2 * t)]
        u = float(random_uniform(1, seed=seed, stream=2 * t + 1)[0])
        xn, pn = list(x), list(p0)
        for _ in range(n_leapfrog):
            xn, pn = _leapfrog(grad, xn, pn, eps)
        ln = logp(xn)
        h0 = -lp + 0.5 * ssum(v * v for v in p0)
        h1 = -ln + 0.5 * ssum(v * v for v in pn)
        if math.log(u) < h0 - h1:
            x, lp = xn, ln
            acc += 1
        out.append(list(x))
    return RichResult(payload={"samples": out, "acceptance": acc / n_iter, "logp": lp})


def nuts_spatial(logp, grad, x0, n_iter: int, *, eps: float = 0.05, max_depth: int = 8, seed: int = 0) -> RichResult:
    r"""No-U-Turn sampler, the efficient slice version (Hoffman and Gelman 2014, Algorithm 3).

    Iteration ``t`` draws the momentum from Philox stream ``2t`` and takes
    its uniforms, in order of use (slice variable, then each direction and
    each subtree acceptance), from a block on stream ``2t + 1``. Trees double
    until a U-turn ``(x+ - x-) . p < 0`` at either end, a divergence
    (``log u - H > 1000``) or ``max_depth``.

    References
    ----------
    Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn sampler: adaptively
    setting path lengths in Hamiltonian Monte Carlo. JMLR 15, 1593-1623.

    Examples
    --------
    >>> r = nuts_spatial(lambda x: -0.5 * x[0] ** 2, lambda x: [-x[0]], [0.5], 10, eps=0.3)
    >>> len(r.samples)
    10
    """
    x = [float(v) for v in x0]
    d = len(x)
    out, depths = [], []
    nblock = 2**max_depth + 2 * max_depth + 4
    for t in range(n_iter):
        p0 = [float(v) for v in random_normal(d, seed=seed, stream=2 * t)]
        U = [float(v) for v in random_uniform(nblock, seed=seed, stream=2 * t + 1)]
        ptr = [0]

        def draw(U=U, ptr=ptr):
            v = U[ptr[0]]
            ptr[0] += 1
            return v

        joint0 = logp(x) - 0.5 * ssum(v * v for v in p0)
        logu = joint0 + math.log(draw())
        xm, xp, pm, pp = list(x), list(x), list(p0), list(p0)
        j, n, s = 0, 1, 1
        xnew = list(x)

        def build(xx, pp_, lu, v, jj):
            if jj == 0:
                x1, p1 = _leapfrog(grad, xx, pp_, v * eps)
                h = logp(x1) - 0.5 * ssum(a * a for a in p1)
                n1 = 1 if lu <= h else 0
                s1 = 1 if lu < h + 1000.0 else 0
                return x1, p1, x1, p1, x1, n1, s1
            xm_, pm_, xp_, pp2, x1, n1, s1 = build(xx, pp_, lu, v, jj - 1)
            if s1 == 1:
                if v == -1:
                    xm_, pm_, _, _, x2, n2, s2 = build(xm_, pm_, lu, v, jj - 1)
                else:
                    _, _, xp_, pp2, x2, n2, s2 = build(xp_, pp2, lu, v, jj - 1)
                if n1 + n2 > 0 and draw() < n2 / (n1 + n2):
                    x1 = x2
                dx = [a - b for a, b in zip(xp_, xm_)]
                s1 = (
                    s2
                    * (1 if ssum(a * b for a, b in zip(dx, pm_)) >= 0 else 0)
                    * (1 if ssum(a * b for a, b in zip(dx, pp2)) >= 0 else 0)
                )
                n1 = n1 + n2
            return xm_, pm_, xp_, pp2, x1, n1, s1

        while s == 1 and j < max_depth:
            v = -1 if draw() < 0.5 else 1
            if v == -1:
                xm, pm, _, _, x1, n1, s1 = build(xm, pm, logu, v, j)
            else:
                _, _, xp, pp, x1, n1, s1 = build(xp, pp, logu, v, j)
            if s1 == 1 and draw() < min(1.0, n1 / n):
                xnew = x1
            n += n1
            dx = [a - b for a, b in zip(xp, xm)]
            s = (
                s1
                * (1 if ssum(a * b for a, b in zip(dx, pm)) >= 0 else 0)
                * (1 if ssum(a * b for a, b in zip(dx, pp)) >= 0 else 0)
            )
            j += 1
        x = list(xnew)
        out.append(list(x))
        depths.append(j)
    return RichResult(payload={"samples": out, "depths": depths})


def tempered_spatial(logp, x0, n_iter: int, temps, *, step: float = 0.1, seed: int = 0) -> RichResult:
    r"""Parallel tempering (replica exchange) with random-walk Metropolis moves.

    Chain ``c`` targets ``pi^(1 / T_c)``. Iteration ``t`` updates every chain
    by a joint Gaussian random walk (step scaled by ``sqrt(T_c)``) and then
    proposes swapping chains ``k = t mod (C - 1)`` and ``k + 1`` with
    probability ``min(1, exp((1/T_k - 1/T_(k+1)) (logp(x_(k+1)) - logp(x_k))))``.
    Streams: base ``t (2C + 1)``, chain ``c`` normals at ``base + 2c``,
    uniforms at ``base + 2c + 1``, swap uniform at ``base + 2C``.

    References
    ----------
    Geyer, C. J. (1991). Markov chain Monte Carlo maximum likelihood.
    Computing Science and Statistics 23, 156-163. Swendsen, R. H. and Wang,
    J.-S. (1986). Phys. Rev. Lett. 57, 2607-2609.

    Examples
    --------
    >>> r = tempered_spatial(lambda x: -0.5 * x[0] ** 2, [0.0], 30, [1.0, 2.0, 4.0], step=1.0)
    >>> len(r.samples), len(r.swap_rate)
    (30, 2)
    """
    C = len(temps)
    xs = [[float(v) for v in x0] for _ in range(C)]
    lps = [logp(x) for x in xs]
    d = len(xs[0])
    out = []
    sw_acc = [0] * (C - 1)
    sw_try = [0] * (C - 1)
    for t in range(n_iter):
        base = t * (2 * C + 1)
        for c in range(C):
            z = random_normal(d, seed=seed, stream=base + 2 * c)
            u = float(random_uniform(1, seed=seed, stream=base + 2 * c + 1)[0])
            prop = [a + step * math.sqrt(temps[c]) * float(b) for a, b in zip(xs[c], z)]
            lq = logp(prop)
            if math.log(u) < (lq - lps[c]) / temps[c]:
                xs[c], lps[c] = prop, lq
        if C > 1:
            k = t % (C - 1)
            u = float(random_uniform(1, seed=seed, stream=base + 2 * C)[0])
            sw_try[k] += 1
            if math.log(u) < (1.0 / temps[k] - 1.0 / temps[k + 1]) * (lps[k + 1] - lps[k]):
                xs[k], xs[k + 1] = xs[k + 1], xs[k]
                lps[k], lps[k + 1] = lps[k + 1], lps[k]
                sw_acc[k] += 1
        out.append(list(xs[0]))
    return RichResult(
        payload={"samples": out, "swap_rate": [a / b if b else 0.0 for a, b in zip(sw_acc, sw_try)], "states": xs}
    )


def cheatsheet() -> str:
    return "car_poisson_target / mh_spatial / hmc_spatial / nuts_spatial / tempered_spatial -> spatial MCMC samplers."
