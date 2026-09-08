# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian re-estimation of taxonomic abundance (Bracken).

Lu J, Breitwieser FP, Thielen P, Salzberg SL (2017), *Bracken:
estimating species abundance in metagenomics data*, PeerJ Computer
Science 3:e104.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["abundance_estimation", "kmer_distribution_from_assignments"]

_METHOD = "Bracken Bayesian abundance re-estimation"


def kmer_distribution_from_assignments(assignment_counts):
    """Column-normalise a matrix of node-by-species read assignments.

    ``assignment_counts[i, j]`` is the number of reads *from* species
    ``j`` that the classifier places at node ``i``. Normalising each
    column gives :math:`P(\\text{node } i \\mid \\text{species } j)`,
    which is what :func:`abundance_estimation` consumes.
    """
    A = np.atleast_2d(np.asarray(assignment_counts, dtype=float))
    if np.any(A < 0):
        raise ValueError("assignment counts must be non-negative.")
    col = A.sum(axis=0)
    if np.any(col <= 0):
        bad = np.flatnonzero(col <= 0).tolist()
        raise ValueError(
            f"species {bad} have no assigned reads, so their k-mer "
            "distribution is undefined."
        )
    return A / col


def abundance_estimation(kraken_output, kmer_distribution, max_iter=1000,
                         tol=1e-12, threshold=0.0):
    r"""Push reads stranded at higher taxonomic ranks down to species.

    A k-mer classifier assigns each read to the lowest node whose k-mers
    identify it uniquely. When two species share the k-mers a read
    carries, the read stops at their common ancestor -- correctly, since
    the read genuinely does not distinguish them. The consequence is
    that many reads never reach species level, and **reading species
    counts straight off the classifier understates every species by a
    different amount**, the amount depending on how much genome each
    shares with its neighbours. That is a systematically distorted
    ranking, not a noisy one.

    Bracken redistributes the stranded reads using the probability that
    a read from species :math:`j` is classified at node :math:`i`:

    .. math::
        P(j \mid i) = \frac{P(i \mid j)\,\theta_j}
                           {\sum_k P(i \mid k)\,\theta_k},

    with :math:`\theta` the species abundance. Since :math:`\theta` is
    what is being estimated, the two steps alternate to a fixed point,
    which is expectation-maximisation for a mixture over species.

    What this cannot do is invent information. If two species have
    identical k-mer profiles their columns of ``kmer_distribution``
    coincide, the likelihood is flat along the direction that trades one
    for the other, and the split between them is decided entirely by the
    starting point. ``identifiable`` reports that rather than returning
    an arbitrary point on a ridge as though it were an estimate.

    Parameters
    ----------
    kraken_output : array-like, shape (n_nodes,)
        Reads assigned to each classification node.
    kmer_distribution : array-like, shape (n_nodes, n_species)
        :math:`P(\text{node} \mid \text{species})`; columns sum to 1.
        See :func:`kmer_distribution_from_assignments`.
    max_iter, tol : int, float
        Expectation-maximisation controls.
    threshold : float
        Species below this many estimated reads are counted in
        ``n_filtered`` rather than silently dropped.

    Returns
    -------
    RichResult
        ``estimate`` (reads per species), ``fractions``,
        ``reads_reassigned``, ``iterations``, ``converged``,
        ``identifiable``, ``naive_species_reads``, ``log_likelihood``.

    References
    ----------
    Lu J, Breitwieser FP, Thielen P, Salzberg SL (2017)
    *PeerJ Comput Sci* 3:e104, doi:10.7717/peerj-cs.104.

    Examples
    --------
    >>> import numpy as np
    >>> # two species; the third node is a shared ancestor
    >>> P = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    >>> reads = np.array([100.0, 300.0, 400.0])
    >>> out = abundance_estimation(reads, P)
    >>> [round(v) for v in out["estimate"]]
    [200, 600]
    """
    r = np.asarray(kraken_output, dtype=float).ravel()
    P = np.atleast_2d(np.asarray(kmer_distribution, dtype=float))
    if P.shape[0] != r.size:
        if P.shape[1] == r.size:
            P = P.T
        else:
            raise ValueError(
                f"kmer_distribution has shape {P.shape} but there are "
                f"{r.size} classification nodes."
            )
    n_nodes, n_sp = P.shape
    if n_sp < 1:
        raise ValueError("need at least one species.")
    if np.any(r < 0):
        raise ValueError("read counts must be non-negative.")
    if np.any(P < 0):
        raise ValueError("k-mer probabilities must be non-negative.")
    colsum = P.sum(axis=0)
    if not np.allclose(colsum, 1.0, atol=1e-6):
        raise ValueError(
            "each column of kmer_distribution must sum to 1; got column "
            f"sums ranging {colsum.min():.6g} to {colsum.max():.6g}."
        )
    total = float(r.sum())
    if total <= 0:
        raise ValueError("no reads to distribute.")

    # a node no species can produce cannot be redistributed to any
    reachable = P.sum(axis=1) > 0
    stranded = float(r[~reachable].sum())

    theta = np.full(n_sp, 1.0 / n_sp)
    it = 0
    converged = False
    r_use = np.where(reachable, r, 0.0)
    for it in range(1, int(max_iter) + 1):
        w = P * theta[None, :]
        den = w.sum(axis=1)
        den_safe = np.where(den > 0, den, 1.0)
        post = w / den_safe[:, None]
        counts = post.T @ r_use
        new = counts / max(counts.sum(), 1e-300)
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            converged = True
            break
        theta = new

    w = P * theta[None, :]
    den = w.sum(axis=1)
    ok = reachable & (den > 0)
    loglik = float(np.sum(r[ok] * np.log(den[ok])))

    usable = total - stranded
    est = theta * usable
    # what the classifier placed at species-unique nodes, for contrast
    unique_node = (P > 0).sum(axis=1) == 1
    naive = np.zeros(n_sp)
    for i in np.flatnonzero(unique_node & reachable):
        naive[int(np.argmax(P[i]))] += r[i]

    # identifiability: duplicate columns leave the likelihood flat
    gram = P.T @ P
    norms = np.sqrt(np.maximum(np.diag(gram), 1e-300))
    cosine = gram / np.outer(norms, norms)
    np.fill_diagonal(cosine, 0.0)
    worst = float(np.max(cosine)) if n_sp > 1 else 0.0
    identifiable = worst < 1.0 - 1e-9

    filtered = int(np.sum(est < threshold)) if threshold > 0 else 0
    out = RichResult(
        title="Bracken abundance re-estimation",
        summary_lines=[
            ("Total reads", total),
            ("Reads redistributed", usable - float(naive.sum())),
            ("Species", n_sp),
            ("Iterations", it),
        ],
        payload={
            "estimate": est,
            "fractions": theta,
            "naive_species_reads": naive,
            "reads_reassigned": float(usable - naive.sum()),
            "reads_unassignable": stranded,
            "total_reads": total,
            "iterations": it,
            "converged": converged,
            "log_likelihood": loglik,
            "identifiable": identifiable,
            "max_column_cosine": worst,
            "n_filtered": filtered,
            "n_nodes": n_nodes,
            "n": n_sp,
            "method": _METHOD,
        },
        interpretation=(
            f"{usable - float(naive.sum()):.0f} of {total:.0f} reads sat "
            "above species level and were redistributed by k-mer "
            "compatibility rather than discarded."
        ),
    )
    if not converged:
        out.warnings.append(
            f"Expectation-maximisation did not converge in {max_iter} "
            "iterations, so the abundances are not a fixed point."
        )
    if not identifiable:
        out.warnings.append(
            f"Two species have indistinguishable k-mer distributions "
            f"(cosine {worst:.6f}). The likelihood is flat along the "
            "direction trading one for the other, so the split between them "
            "reflects the starting point, not the data."
        )
    if stranded > 0:
        out.warnings.append(
            f"{stranded:.0f} reads sit at nodes no species in the "
            "distribution can produce. They are excluded from the total "
            "rather than spread over species that cannot explain them."
        )
    return out


def cheatsheet():
    return (
        "abndst: Bracken expectation-maximisation pushing reads stranded at "
        "higher taxonomic ranks down to species by k-mer compatibility"
    )
