# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Hamiltonian Monte Carlo.

Neal (2011), "MCMC using Hamiltonian dynamics", in Brooks, Gelman,
Jones and Meng (eds), *Handbook of Markov Chain Monte Carlo*, CRC
Press, chapter 5, doi:10.1201/b10905.  The state is augmented with a
momentum, H(x, p) = U(x) + K(p) with U = -log target and K = p'p/2,
and the leapfrog integrator

    p <- p - (eps/2) grad U(x),
    x <- x + eps p,
    p <- p - (eps/2) grad U(x)

is run L times before a Metropolis accept/reject on exp(-Delta H).
Leapfrog is exactly reversible and volume preserving, which is what
makes the acceptance ratio simply exp(-Delta H); both properties are
checked directly in the tests.  Momenta and acceptance uniforms come
from the deterministic van der Corput / inverse-normal streams so the
chain is reproducible.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hamiltonian_mc"]


def _leapfrog(grad_log_p, x, p, eps, L):
    x = list(x)
    p = list(p)
    d = len(x)
    g = core.vec(grad_log_p(x))
    for _ in range(L):
        for j in range(d):
            p[j] += 0.5 * eps * g[j]
        for j in range(d):
            x[j] += eps * p[j]
        g = core.vec(grad_log_p(x))
        for j in range(d):
            p[j] += 0.5 * eps * g[j]
    return x, p


def hamiltonian_mc(log_p, grad_log_p, x0, step_size=0.1, L=10, n_iter=200):
    """Run HMC and return the chain summaries."""
    x = core.vec(x0)
    d = len(x)
    if d == 0:
        raise ValueError("hamiltonian_mc: x0 is empty")
    if not callable(log_p) or not callable(grad_log_p):
        raise ValueError("hamiltonian_mc: log_p and grad_log_p must be callable")
    eps = float(step_size)
    if eps <= 0:
        raise ValueError("hamiltonian_mc: step_size must be positive")
    steps = int(L)
    if steps < 1:
        raise ValueError("hamiltonian_mc: L must be at least 1")
    it = int(n_iter)
    if it < 1:
        raise ValueError("hamiltonian_mc: n_iter must be at least 1")
    draws = []
    acc = 0
    counter = 1
    dH = []
    for _ in range(it):
        p0 = [core.qnorm(core.vdc(counter + j, 2)) for j in range(d)]
        counter += d
        H0 = -float(log_p(x)) + 0.5 * sum(v * v for v in p0)
        xn, pn = _leapfrog(grad_log_p, x, p0, eps, steps)
        H1 = -float(log_p(xn)) + 0.5 * sum(v * v for v in pn)
        dH.append(H1 - H0)
        u = core.vdc(counter, 3)
        counter += 1
        accept = True if H1 <= H0 else (u < math.exp(-(H1 - H0)))
        if accept:
            x = xn
            acc += 1
        draws.append(list(x))
    means = [sum(row[j] for row in draws) / it for j in range(d)]
    return RichResult(
        title="Hamiltonian Monte Carlo",
        summary_lines=[("draws", it), ("acceptance", acc / it)],
        payload={
            "estimate": means[0],
            "mean": means,
            "draws": draws,
            "accept_rate": acc / float(it),
            "mean_energy_error": sum(dH) / len(dH),
            "n": it,
            "method": "leapfrog integration of H = U + K with a Metropolis correction, Neal (2011) ch. 5",
        },
    )


def cheatsheet():
    return "hmcsam: Hamiltonian Monte Carlo"
