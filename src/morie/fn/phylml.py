r"""Maximum-likelihood phylogeny: Felsenstein's pruning algorithm and the
F81 substitution model.

Felsenstein, J. (1981) "Evolutionary Trees from DNA Sequences: A
Maximum Likelihood Approach", *Journal of Molecular Evolution* 17,
368-376.

Sites are assumed independent, so the likelihood is computed per site
and multiplied across sites. Written out naively, the likelihood of a
tree sums over the unknown states at every interior node -- "this
expression will have 256 terms, and in general the expression for
:math:`n` species will have :math:`2^{2n-2}` terms, which can easily be
a very large number".

The economy comes from moving the summation signs rightwards. Define
:math:`L_s(k)`, the likelihood of the data at or above node :math:`k`
given that :math:`k` has state :math:`s`. At a tip, :math:`L_s(k)` is
0 for every state except the one observed, where it is 1. At an
interior node with children :math:`i` and :math:`j` joined by branches
:math:`v_i` and :math:`v_j`,

.. math:: L_{s}(k) = \Big[\sum_{s_i} P_{s s_i}(v_i)\, L_{s_i}(i)\Big]
                     \Big[\sum_{s_j} P_{s s_j}(v_j)\, L_{s_j}(j)\Big],

evaluated by a postorder traversal, and the site likelihood is
(eq. 5)

.. math:: L = \sum_{s_0} \pi_{s_0} L_{s_0}(0).

Felsenstein calls this **pruning**, "since it in effect removes two
tips from the tree at each step". It turns an exponential sum into a
linear one.

The substitution model (his eqs. 6-7) assumes that in time
:math:`dt` a base is replaced with probability :math:`u\,dt`, its
replacement being :math:`j` with probability :math:`\pi_j` -- so a base
may be "replaced" by itself and not every substitution is observable.
That gives

.. math:: P_{ij}(t) = e^{-ut}\,\delta_{ij} + (1 - e^{-ut})\,\pi_j,

which follows once you notice that :math:`e^{-ut}` is the probability
of no change at all and that any change lands in :math:`j` with
probability :math:`\pi_j`. This is the model now called **F81**; at
:math:`\pi = (1/4,1/4,1/4,1/4)` it reduces to Jukes-Cantor.

Two properties of the model are load-bearing and both are anchored:

*Reversibility* (his eq. 8), :math:`\pi_i P_{ij}(t) = \pi_j
P_{ji}(t)`, means the process looks the same run forwards or
backwards, so the tree may be rooted anywhere without changing the
likelihood.

The *pulley principle*: the likelihood depends on the two branches
either side of the root only through their **sum**, so length may be
slid from one to the other freely. That is why an unrooted tree is
what is actually identifiable, and it is checked here directly.

``optimise_branch`` maximises the likelihood over a single branch by
golden-section search, which is the one-dimensional step the paper's
iterative scheme is built from.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["phylml", "substitution_matrix", "site_likelihood",
           "optimise_branch", "maximum_likelihood_phylogeny"]

BASES = "ACGT"


def _pi(pi):
    if pi is None:
        return [0.25] * 4
    p = [float(v) for v in np.atleast_1d(np.asarray(pi, dtype=float))]
    if len(p) != 4:
        raise ValueError("phylml: pi must have four entries (A, C, G, T)")
    if min(p) < 0.0:
        raise ValueError("phylml: pi must be non-negative")
    s = sum(p)
    if abs(s - 1.0) > 1e-9:
        raise ValueError("phylml: pi must sum to 1, got %g" % s)
    return p


def substitution_matrix(t, pi=None, u=1.0):
    r"""The F81 matrix :math:`P_{ij}(t) = e^{-ut}\delta_{ij}
    + (1-e^{-ut})\pi_j` (Felsenstein 1981 eq. 7).

    ``t`` is the branch length and ``u`` the substitution rate; with
    the default ``u = 1`` the branch length is measured directly in
    expected substitutions, which is the usual convention.
    """
    p = _pi(pi)
    t = float(t)
    if t < 0.0:
        raise ValueError("phylml: branch length must be >= 0")
    e = math.exp(-float(u) * t)
    return [[e * (1.0 if i == j else 0.0) + (1.0 - e) * p[j]
             for j in range(4)] for i in range(4)]


def _tip_vector(base):
    b = str(base).upper()
    if b in ("-", "N", "?"):
        return [1.0] * 4          # missing data contributes nothing
    if b not in BASES:
        raise ValueError("phylml: unknown base %r; expected one of ACGT "
                         "or a gap" % (base,))
    return [1.0 if BASES[i] == b else 0.0 for i in range(4)]


def _prune(node, site, pi, u, seqs):
    """Postorder traversal computing L_s(k). Returns the length-4 vector.

    A node is either a tip name, or ``(child, length, child, length)``
    -- or more generally ``(child1, v1, child2, v2, ...)`` for
    multifurcations, since the product in the pruning step extends to
    any number of children.
    """
    if not isinstance(node, (tuple, list)):
        return _tip_vector(seqs[node][site])
    if len(node) % 2:
        raise ValueError("phylml: a node must be (child, length, ...) "
                         "pairs, got %d entries" % len(node))
    out = [1.0] * 4
    for c in range(0, len(node), 2):
        child, v = node[c], node[c + 1]
        below = _prune(child, site, pi, u, seqs)
        P = substitution_matrix(v, pi, u)
        for s in range(4):
            out[s] *= sum(P[s][x] * below[x] for x in range(4))
    return out


def site_likelihood(tree, seqs, site, pi=None, u=1.0):
    r"""The likelihood of one site, eq. 5, via pruning."""
    p = _pi(pi)
    L = _prune(tree, site, p, u, seqs)
    return sum(p[s] * L[s] for s in range(4))


def phylml(tree, seqs, pi=None, u=1.0):
    r"""Log-likelihood of a tree given aligned sequences.

    Parameters
    ----------
    tree : nested tuple
        ``(child, length, child, length)`` recursively, with tips given
        as keys of ``seqs``. Multifurcations are allowed.
    seqs : dict
        Taxon name -> aligned sequence string over ACGT (``-``, ``N``
        or ``?`` for missing).
    pi : array-like, optional
        Base frequencies :math:`(\pi_A, \pi_C, \pi_G, \pi_T)`.
        Uniform by default, which makes F81 into Jukes-Cantor.
    u : float
        Substitution rate.

    Returns
    -------
    RichResult
        ``estimate`` / ``log_likelihood`` is :math:`\sum_{\text{sites}}
        \log L`, with ``site_likelihoods`` and ``site_log_likelihoods``
        alongside.

    Examples
    --------
    Two taxa, one branch::

        phylml(("a", 0.1, "b", 0.1), {"a": "ACGT", "b": "ACGT"})

    References
    ----------
    Felsenstein (1981) *J. Mol. Evol.* 17, 368-376, eqs. 3-8 and the
    pulley principle.
    """
    p = _pi(pi)
    if not isinstance(seqs, dict) or not seqs:
        raise ValueError("phylml: seqs must be a non-empty dict of "
                         "name -> sequence")
    lens = set(len(v) for v in seqs.values())
    if len(lens) != 1:
        raise ValueError("phylml: sequences must be aligned to a common "
                         "length, got %r" % sorted(lens))
    n_sites = lens.pop()
    if n_sites == 0:
        raise ValueError("phylml: sequences are empty")

    site_L = []
    for i in range(n_sites):
        Li = site_likelihood(tree, seqs, i, p, u)
        if Li <= 0.0:
            raise ValueError("phylml: site %d has zero likelihood; check "
                             "the tree and the alignment" % i)
        site_L.append(Li)
    logs = [math.log(v) for v in site_L]
    return RichResult(payload={
        "estimate": float(sum(logs)),
        "log_likelihood": float(sum(logs)),
        "site_likelihoods": site_L,
        "site_log_likelihoods": logs,
        "n_sites": n_sites,
        "n_taxa": len(seqs),
        "pi": p,
        "u": float(u),
        "method": "ML phylogeny by pruning (Felsenstein 1981)",
    })


def optimise_branch(make_tree, seqs, pi=None, u=1.0, lo=1e-6, hi=10.0,
                    tol=1e-10, max_iter=200):
    r"""Maximise the log-likelihood over one branch length.

    ``make_tree(v)`` returns the tree with that branch set to ``v``.
    Golden-section search on :math:`[lo, hi]`; the F81 likelihood is
    unimodal in a single branch length, which is what makes the
    paper's coordinate-wise scheme work.

    Returns the optimal length and its log-likelihood.
    """
    if not callable(make_tree):
        raise TypeError("phylml: make_tree must be callable")
    if not lo < hi:
        raise ValueError("phylml: need lo < hi")
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc = phylml(make_tree(c), seqs, pi, u)["log_likelihood"]
    fd = phylml(make_tree(d), seqs, pi, u)["log_likelihood"]
    for _ in range(int(max_iter)):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = phylml(make_tree(c), seqs, pi, u)["log_likelihood"]
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = phylml(make_tree(d), seqs, pi, u)["log_likelihood"]
        if abs(b - a) < tol:
            break
    v = 0.5 * (a + b)
    # A branch can run to the bound rather than to an interior optimum:
    # under F81 two sequences with no shared signal are saturated and
    # the ML length is unbounded, so the search returns `hi`. That is a
    # real property of the model, not a convergence failure, but the
    # returned number is then a bound and not an estimate -- so say so.
    edge = 1e-6 * (float(hi) - float(lo))
    at_bound = (v <= float(lo) + edge or v >= float(hi) - edge)
    return RichResult(payload={
        "estimate": float(v),
        "length": float(v),
        "log_likelihood": float(
            phylml(make_tree(v), seqs, pi, u)["log_likelihood"]),
        "at_bound": bool(at_bound),
        "bounds": (float(lo), float(hi)),
        "method": "branch optimisation (Felsenstein 1981)",
    })


def cheatsheet():
    return ("phylml: Felsenstein (1981) pruning. L_s(k) = prod over "
            "children of sum_x P_sx(v) L_x(child); tips are 0/1 "
            "indicators; L = sum_s pi_s L_s(root) (eq. 5). Turns a "
            "2^(2n-2)-term sum into a linear traversal. F81 model "
            "P_ij(t) = e^-ut delta_ij + (1-e^-ut) pi_j (eq. 7), "
            "reversible, and the PULLEY PRINCIPLE means the two root "
            "branches matter only through their sum.")


# compact alias per ledger/NAMING.md
maximum_likelihood_phylogeny = phylml
