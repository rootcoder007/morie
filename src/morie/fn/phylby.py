# morie.fn -- function file (rootcoder007/morie)
r"""MrBayes: Bayesian phylogenetic inference under mixed models.

Ronquist, F., & Huelsenbeck, J. P. (2003) "MrBayes 3: Bayesian
phylogenetic inference under mixed models", *Bioinformatics* 19(12),
1572-1574. doi:10.1093/bioinformatics/btg180

Where maximum likelihood asks which tree makes the data most probable,
this asks for the *posterior* over trees,

.. math::

   p(\tau, v, \theta \mid X) \;\propto\;
   p(X \mid \tau, v, \theta)\, p(\tau)\, p(v)\, p(\theta),

and samples it by Markov chain Monte Carlo, because the sum over
topologies has no closed form. The likelihood itself is Felsenstein's
pruning recursion and is taken from :mod:`morie.fn.phylml` rather than
rewritten -- what is new here is everything above it.

**Metropolis-coupled MCMC** (the paper's headline feature, "MC^3").
:math:`n` chains run at once, chain :math:`j` at temperature

.. math::

   \beta_j = \frac{1}{1 + \lambda j},

so the cold chain (:math:`j = 0`, :math:`\beta = 1`) samples the true
posterior while the heated ones explore a flattened surface and cross
valleys the cold chain would be stuck behind. Only the cold chain is
recorded. Periodically two chains are proposed for a swap, accepted with

.. math::

   \min\left(1,\;
   \frac{\pi(\theta_k)^{\beta_j}\,\pi(\theta_j)^{\beta_k}}
        {\pi(\theta_j)^{\beta_j}\,\pi(\theta_k)^{\beta_k}}\right)
   = \min\bigl(1, \exp[(\beta_j - \beta_k)(\ell_k - \ell_j)]\bigr),

which is the only place the temperatures enter the acceptance rule.

**Moves.** Topology by nearest-neighbour interchange, branch lengths by
a multiplier proposal (a random walk on the log scale, whose Hastings
ratio is the multiplier itself), and the substitution rate likewise.
Priors follow the paper's defaults: uniform on topologies and
exponential on branch lengths.

**Mixed models.** The paper's other feature is partitioning: different
sites obey different models. ``partitions`` assigns each site to a
partition and each partition carries its own rate multiplier, so a fast
and a slow gene can share a topology while keeping separate rates.

**What convergence is judged by.** Two independent runs are made and the
average standard deviation of split frequencies between them is
reported, which is the diagnostic MrBayes prints; a value near zero
means the two runs are sampling the same posterior.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult
from .phylml import phylml

__all__ = [
    "phylby",
    "bayesian_phylogeny",
    "log_posterior",
    "nni_neighbours",
    "topology_key",
    "splits_of",
    "chain_temperature",
    "swap_acceptance",
    "clade_credibility",
]


def chain_temperature(j, lam=0.2):
    r"""Chain :math:`j`'s heat, :math:`\beta_j = 1/(1 + \lambda j)`."""
    if lam < 0:
        raise ValueError("phylby: the heating parameter must be >= 0")
    if j < 0:
        raise ValueError("phylby: the chain index must be >= 0")
    return 1.0 / (1.0 + lam * float(j))


def swap_acceptance(beta_j, beta_k, logp_j, logp_k):
    r"""Acceptance probability of swapping two chains' states."""
    return min(1.0, math.exp(min((beta_j - beta_k) * (logp_k - logp_j),
                                 700.0)))


def _tips(node, out=None):
    if out is None:
        out = []
    if not isinstance(node, (tuple, list)):
        out.append(node)
        return out
    for c in range(0, len(node), 2):
        _tips(node[c], out)
    return out


def splits_of(tree):
    """The set of splits (bipartitions) an unrooted tree induces."""
    all_tips = sorted(_tips(tree))
    n = len(all_tips)
    out = set()

    def walk(node):
        if not isinstance(node, (tuple, list)):
            return {node}
        below = set()
        for c in range(0, len(node), 2):
            got = walk(node[c])
            below |= got
        if 1 < len(below) < n - 1:
            side = frozenset(below)
            other = frozenset(t for t in all_tips if t not in below)
            out.add(side if sorted(side)[0] < sorted(other)[0] else other)
        return below

    walk(tree)
    return out


def topology_key(tree):
    """A canonical label for the topology, ignoring branch lengths."""
    return tuple(sorted(tuple(sorted(s)) for s in splits_of(tree)))


def _replace_branch(node, path, value):
    """Return a copy of the tree with one branch length replaced."""
    if not path:
        return node
    idx = path[0]
    parts = list(node)
    if len(path) == 1:
        parts[idx] = value
    else:
        parts[idx] = _replace_branch(parts[idx], path[1:], value)
    return tuple(parts)


def _branch_paths(node, path=()):
    """Paths to every branch-length slot in the tree."""
    out = []
    if not isinstance(node, (tuple, list)):
        return out
    if len(node) % 2:
        raise ValueError("phylby: a node must be (child, length) pairs, "
                         "got %d entries" % len(node))
    for c in range(0, len(node), 2):
        out.append(path + (c + 1,))
        out.extend(_branch_paths(node[c], path + (c,)))
    return out


def _subtrees(node, path=()):
    out = []
    if not isinstance(node, (tuple, list)):
        return out
    for c in range(0, len(node), 2):
        out.append((path + (c,), node[c]))
        out.extend(_subtrees(node[c], path + (c,)))
    return out


def _set_at(node, path, value):
    if not path:
        return value
    parts = list(node)
    parts[path[0]] = _set_at(parts[path[0]], path[1:], value)
    return tuple(parts)


def _get_at(node, path):
    for i in path:
        node = node[i]
    return node


def nni_neighbours(tree):
    """Every tree one nearest-neighbour interchange away.

    An NNI swaps a subtree with one of its aunts; the result is a
    different topology on the same tip set.
    """
    out = []
    subs = _subtrees(tree)
    for pa, a in subs:
        for pb, b in subs:
            if pa == pb:
                continue
            if pa[:len(pb)] == pb or pb[:len(pa)] == pa:
                continue          # one contains the other
            cand = _set_at(_set_at(tree, pa, b), pb, a)
            if topology_key(cand) != topology_key(tree):
                out.append(cand)
    seen, uniq = set(), []
    for t in out:
        k = topology_key(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def log_posterior(tree, seqs, pi=None, rate=1.0, branch_prior_mean=0.1,
                  partitions=None, rates=None, temperature=1.0):
    r"""Log posterior up to a constant: likelihood plus the priors.

    The topology prior is uniform, so it contributes a constant and is
    dropped. Branch lengths are exponential with mean
    ``branch_prior_mean``. With ``partitions`` the sites are split and
    each partition scales the rate by its own multiplier.
    """
    if branch_prior_mean <= 0:
        raise ValueError("phylby: the branch prior mean must be positive")
    if rate <= 0:
        raise ValueError("phylby: the substitution rate must be positive")
    lengths = [_get_at(tree, p) for p in _branch_paths(tree)]
    if any(v < 0 for v in lengths):
        raise ValueError("phylby: branch lengths must be non-negative")
    lam = 1.0 / branch_prior_mean
    log_prior = sum(math.log(lam) - lam * v for v in lengths)
    if partitions is None:
        ll = phylml(tree, seqs, pi, rate)["log_likelihood"]
    else:
        names = sorted(set(partitions))
        rates = rates or dict((k, 1.0) for k in names)
        if any(rates.get(k, 1.0) <= 0 for k in names):
            raise ValueError("phylby: partition rates must be positive")
        ll = 0.0
        for k in names:
            keep = [i for i, p in enumerate(partitions) if p == k]
            sub = dict((t, "".join(s[i] for i in keep))
                       for t, s in seqs.items())
            ll += phylml(tree, sub, pi, rate * rates[k])["log_likelihood"]
        log_prior += sum(-rates[k] for k in names)   # exponential(1)
    return {"loglik": ll, "logprior": log_prior,
            "logpost": temperature * (ll + log_prior)}


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def clade_credibility(samples):
    """Posterior probability of each split across the sampled trees."""
    if not samples:
        raise ValueError("phylby: no samples to summarise")
    counts = {}
    for t in samples:
        for s in splits_of(t):
            key = tuple(sorted(s))
            counts[key] = counts.get(key, 0) + 1
    n = float(len(samples))
    return dict((k, v / n) for k, v in counts.items())


def _step(state, seqs, pi, prior_mean, partitions, rnd, beta, tune):
    """One Metropolis-Hastings move on one chain."""
    tree, rate, rates = state["tree"], state["rate"], state["rates"]
    u = rnd()
    hastings = 0.0
    if u < 0.4:                                   # topology: NNI
        cand = nni_neighbours(tree)
        if not cand:
            return state, False
        new = (cand[int(rnd() * len(cand))], rate, rates)
    elif u < 0.8:                                 # branch: multiplier
        paths = _branch_paths(tree)
        p = paths[int(rnd() * len(paths))]
        m = math.exp(tune * (rnd() - 0.5))
        new_tree = _replace_branch(tree, p, _get_at(tree, p) * m)
        hastings = math.log(m)                    # Jacobian of log-scale
        new = (new_tree, rate, rates)
    else:                                         # rate: multiplier
        m = math.exp(tune * (rnd() - 0.5))
        hastings = math.log(m)
        new = (tree, rate * m, rates)
    cur = log_posterior(tree, seqs, pi, rate, prior_mean, partitions,
                        rates)["logpost"]
    prop = log_posterior(new[0], seqs, pi, new[1], prior_mean, partitions,
                         new[2])["logpost"]
    logalpha = beta * (prop - cur) + hastings
    if math.log(max(rnd(), 1e-300)) < logalpha:
        return {"tree": new[0], "rate": new[1], "rates": new[2]}, True
    return state, False


def phylby(alignment, n_iter=2000, burnin=None, n_chains=4, lam=0.2,
           swap_every=10, sample_every=10, pi=None, rate=1.0,
           branch_prior_mean=0.1, partitions=None, tree=None,
           n_runs=2, tune=1.0, seed=0):
    """Sample the posterior over trees (Ronquist & Huelsenbeck 2003).

    ``alignment`` maps taxon name to sequence. Returns the sampled cold
    chain, clade credibilities and the between-run split-frequency
    diagnostic.
    """
    seqs = dict((str(k), str(v).upper()) for k, v in alignment.items())
    if len(seqs) < 4:
        raise ValueError("phylby: at least four taxa are needed for an "
                         "unrooted topology to vary")
    L = set(len(v) for v in seqs.values())
    if len(L) != 1 or L == {0}:
        raise ValueError("phylby: sequences must be aligned and non-empty")
    if n_iter < 1 or n_chains < 1 or n_runs < 1:
        raise ValueError("phylby: n_iter, n_chains and n_runs must be "
                         "positive")
    if swap_every < 1 or sample_every < 1:
        raise ValueError("phylby: swap_every and sample_every must be "
                         "positive")
    if partitions is not None and len(partitions) != len(next(
            iter(seqs.values()))):
        raise ValueError("phylby: one partition label per site is required")
    burn = n_iter // 2 if burnin is None else int(burnin)
    if not 0 <= burn < n_iter:
        raise ValueError("phylby: burnin must be less than n_iter")
    names = sorted(seqs)
    if tree is None:
        tree = (names[0], 0.1, names[1], 0.1)
        for t in names[2:]:
            tree = (tree, 0.1, t, 0.1)
    parts = sorted(set(partitions)) if partitions else []
    runs, accepted, swaps, proposed_swaps = [], 0, 0, 0
    for r in range(int(n_runs)):
        rnd = _rng(seed + 1000 * r + 1)
        chains = [{"tree": tree, "rate": rate,
                   "rates": dict((k, 1.0) for k in parts)}
                  for _ in range(int(n_chains))]
        betas = [chain_temperature(j, lam) for j in range(int(n_chains))]
        samples = []
        for it in range(int(n_iter)):
            for j in range(int(n_chains)):
                chains[j], ok = _step(chains[j], seqs, pi,
                                      branch_prior_mean, partitions, rnd,
                                      betas[j], tune)
                if j == 0 and ok:
                    accepted += 1
            if int(n_chains) > 1 and (it + 1) % int(swap_every) == 0:
                a = int(rnd() * n_chains)
                b = int(rnd() * n_chains)
                if a != b:
                    proposed_swaps += 1
                    la = log_posterior(chains[a]["tree"], seqs, pi,
                                       chains[a]["rate"],
                                       branch_prior_mean, partitions,
                                       chains[a]["rates"])["logpost"]
                    lb = log_posterior(chains[b]["tree"], seqs, pi,
                                       chains[b]["rate"],
                                       branch_prior_mean, partitions,
                                       chains[b]["rates"])["logpost"]
                    if rnd() < swap_acceptance(betas[a], betas[b], la, lb):
                        chains[a], chains[b] = chains[b], chains[a]
                        swaps += 1
            if it >= burn and (it - burn) % int(sample_every) == 0:
                samples.append(chains[0]["tree"])
        if not samples:
            samples = [chains[0]["tree"]]
        runs.append(samples)

    cred = [clade_credibility(s) for s in runs]
    keys = set()
    for c in cred:
        keys |= set(c)
    asdsf = 0.0
    if len(runs) > 1 and keys:
        tot = 0.0
        for k in keys:
            fs = [c.get(k, 0.0) for c in cred]
            m = sum(fs) / len(fs)
            tot += math.sqrt(sum((f - m) ** 2 for f in fs) / len(fs))
        asdsf = tot / len(keys)
    pooled = [t for s in runs for t in s]
    topo = {}
    for t in pooled:
        k = topology_key(t)
        topo[k] = topo.get(k, 0) + 1
    best = max(topo, key=lambda k: topo[k])
    return RichResult(payload={
        "estimate": clade_credibility(pooled),
        "clade_credibility": clade_credibility(pooled),
        "samples": pooled,
        "runs": runs,
        "map_topology": best,
        "map_probability": topo[best] / float(len(pooled)),
        "topology_counts": topo,
        "asdsf": asdsf,
        "acceptance": accepted / float(n_iter * n_runs),
        "swap_rate": (swaps / float(proposed_swaps)) if proposed_swaps
                     else 0.0,
        "temperatures": [chain_temperature(j, lam)
                         for j in range(int(n_chains))],
        "n_chains": int(n_chains),
        "n_runs": int(n_runs),
        "n_samples": len(pooled),
        "method": ("MrBayes 3 (Ronquist & Huelsenbeck 2003): "
                   "Metropolis-coupled MCMC over topology, branch "
                   "lengths and model, with a uniform topology prior and "
                   "exponential branch lengths"),
        "note": ("the likelihood is Felsenstein pruning from "
                 "morie.fn.phylml; asdsf is the average standard "
                 "deviation of split frequencies between independent "
                 "runs, the diagnostic MrBayes prints, and should "
                 "approach zero"),
    })


bayesian_phylogeny = phylby


def cheatsheet():
    return ("phylby: MrBayes 3 (Ronquist & Huelsenbeck 2003). MCMC over "
            "(topology, branch lengths, rate) with a uniform topology "
            "prior and exponential branch lengths, the likelihood coming "
            "from Felsenstein pruning. Metropolis coupling runs n chains "
            "at beta_j = 1/(1 + lambda j) and swaps them with "
            "min(1, exp[(beta_j - beta_k)(l_k - l_j)]); only the cold "
            "chain is sampled. Partitions give each subset of sites its "
            "own rate. Convergence is judged by the average standard "
            "deviation of split frequencies between independent runs.")
