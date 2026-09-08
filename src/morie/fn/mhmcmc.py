# morie.fn -- tail3 batch (rootcoder007/morie)
"""Metropolis-Hastings sampler.

Sources consulted: Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N.,
Teller, A.H. & Teller, E. (1953). Equation of state calculations by fast
computing machines.  *Journal of Chemical Physics* 21(6), 1087-1092;
Hastings, W.K. (1970). Monte Carlo sampling methods using Markov chains and
their applications.  *Biometrika* 57(1), 97-109.  Hastings generalises the
1953 symmetric-proposal rule to an arbitrary proposal q, giving the
acceptance probability

    alpha(x -> x') = min( 1,  pi(x') q(x | x') / ( pi(x) q(x' | x) ) )

which reduces to min(1, pi(x')/pi(x)) when q is symmetric, the Metropolis
et al. (1953) form.  A Gaussian random walk is used by default, so ``q`` may
be left unset.

Both the proposal increments ``z`` and the acceptance uniforms ``u`` are
supplied by the caller, so a chain is exactly reproducible and the R mirror
follows the identical path.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["metropolis_hastings"]


def metropolis_hastings(target, x0=0.0, n_iter=1000, u=None, z=None, scale=1.0, q=None, burn=0):
    """Run a scalar Metropolis-Hastings chain with caller-supplied noise.

    Parameters
    ----------
    target : callable
        Unnormalised target density pi(x), returning a non-negative number.
    x0 : float
        Starting state.
    n_iter : int
        Number of iterations.
    u : array-like
        ``n_iter`` uniforms in [0, 1) used for the accept/reject decision.
    z : array-like
        ``n_iter`` standard normal increments for the random-walk proposal.
    scale : float
        Random-walk proposal standard deviation.
    q : callable, optional
        Proposal density ``q(x_to, x_from)``.  If omitted the proposal is
        treated as symmetric and the Hastings ratio drops out.
    burn : int
        Number of leading draws to discard before summarising.

    Returns
    -------
    RichResult
        estimate (posterior mean), sd, accept_rate, chain, accepted, n, method.

    References
    ----------
    Metropolis et al. (1953); Hastings (1970), Biometrika 57(1), 97-109.
    """
    ni = int(n_iter)
    uv = [float(v) for v in np.atleast_1d(np.asarray(u, dtype=float)).ravel()]
    zv = [float(v) for v in np.atleast_1d(np.asarray(z, dtype=float)).ravel()]
    x = float(x0)
    px = float(target(x))
    chain = []
    accepted = 0
    for i in range(ni):
        prop = x + float(scale) * zv[i % len(zv)]
        pp = float(target(prop))
        if px <= 0.0:
            ratio = 1.0 if pp > 0.0 else 0.0
        else:
            ratio = pp / px
        if q is not None:
            qf = float(q(prop, x))
            qb = float(q(x, prop))
            ratio = ratio * (qb / qf) if qf > 0.0 else 0.0
        alpha = ratio if ratio < 1.0 else 1.0
        if uv[i % len(uv)] < alpha:
            x = prop
            px = pp
            accepted += 1
        chain.append(x)
    keep = chain[int(burn):] if int(burn) < ni else chain
    arr = np.asarray(keep, dtype=float)
    m = float(np.mean(arr))
    k = int(arr.size)
    sd = float(np.sqrt(sum((float(v) - m) ** 2 for v in keep) / (k - 1))) if k > 1 else float("nan")
    return RichResult(
        payload={
            "estimate": m,
            "sd": sd,
            "accept_rate": float(accepted) / ni if ni > 0 else float("nan"),
            "chain": np.asarray(chain, dtype=float),
            "accepted": accepted,
            "n": ni,
            "method": "Metropolis-Hastings (Metropolis et al. 1953; Hastings 1970)",
        }
    )


# CANONICAL TEST
# >>> # a flat target accepts every proposal
# >>> u = [0.5] * 10
# >>> z = [1.0] * 10
# >>> r = metropolis_hastings(lambda x: 1.0, 0.0, 10, u=u, z=z, scale=1.0)
# >>> assert r["accepted"] == 10
# >>> assert abs(float(r["chain"][9]) - 10.0) < 1e-12


def cheatsheet():
    return "mhmcmc(target, x0, n_iter, u, z): Metropolis-Hastings chain."
