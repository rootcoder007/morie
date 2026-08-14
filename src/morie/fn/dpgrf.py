# morie.fn -- function file (rootcoder007/morie)
r"""Difference boundaries in areal data: let the prior make ties.

Areal wombling asks where the map *breaks* -- which pairs of adjacent
regions differ enough to draw a boundary between them. The obvious
approach fails for a reason worth stating precisely.

**A continuous prior can never say two regions are the same.** Give
each region a random effect from a CAR model and the posterior
probability that two of them are *equal* is zero, because a
continuous distribution assigns no mass to a point. So "is there a
boundary here?" collapses into "is this difference large?", and the
answer depends entirely on a threshold nobody can justify.

**A Dirichlet process prior puts positive mass on ties.** Under
:math:`\phi_i \sim G`, :math:`G \sim DP(\alpha, G_0)`, the Pólya urn
gives repeats positive probability, so regions genuinely **cluster**
and the posterior can report :math:`P(\phi_i \ne \phi_j)` for
adjacent :math:`i, j` as an honest probability. A boundary is then a
posterior statement about cluster membership, not a threshold on a
difference. That is the whole argument for the DP here, and it is why
``boundary_probabilities`` is the output that matters.

**The spatial structure still belongs in the model.** The DP handles
the ties; the CAR component handles the fact that neighbouring
regions are a priori similar -- through the adjacency matrix and its
row sums. Dropping the CAR part leaves an exchangeable clustering
that has forgotten the map, and ``car_precision`` builds
:math:`D - \rho W` so the two roles stay separate.

**Co-clustering is the summary, and it must be symmetric.** The
posterior probability that two regions share a cluster is estimated
from the sampled labels; ``coclustering`` returns the full matrix,
which is symmetric with a unit diagonal by construction -- a cheap
invariant that catches an indexing error immediately.

References
----------
Li, P., Banerjee, S., Hanson, T. A. & McBean, A. M. (2015) "Bayesian
Models for Detecting Difference Boundaries in Areal Data",
*Statistica Sinica* 25(1), 385-402, doi:10.5705/ss.2013.238w. [PDF
supplied by Vee; HHS Public Access author manuscript.] Areal data,
conditional autoregressive models, difference boundaries and
wombling; the Dirichlet process mixture prior on the areal random
effects, with G ~ DP(alpha, G_0); the stick-breaking representation
of the DP (Sethuraman, 1994) and the Blackwell-MacQueen generalized
Polya urn scheme giving an explicit prediction rule and effective
sampling strategies; and the detection of difference boundaries
through the induced clustering of neighbouring regions.

Sethuraman, J. (1994) "A Constructive Definition of Dirichlet
Priors", *Statistica Sinica* 4(2), 639-650. The stick-breaking
representation; implemented in :mod:`slowdp`.

Besag, J. (1974) "Spatial Interaction and the Statistical Analysis of
Lattice Systems", *JRSS-B* 36(2), 192-236,
doi:10.1111/j.2517-6161.1974.tb00999.x. The conditional
autoregressive model.

Womble, W. H. (1951) "Differential Systematics", *Science*
114(2961), 315-322, doi:10.1126/science.114.2961.315. The boundary
problem the method is named for.
"""

import math

from . import _array_core as np
from . import _s03core as k
from . import posspr as urn
from ._richresult import RichResult

__all__ = ["adjacency_pairs", "car_precision", "sample_labels",
           "coclustering", "boundary_probabilities",
           "continuous_prior_tie_probability"]

_EPS = 1e-12


def adjacency_pairs(W):
    r"""The adjacent pairs -- the only ones a boundary can sit
    between."""
    A = [[float(v) for v in r] for r in k.mat(W)]
    n = len(A)
    if any(len(r) != n for r in A):
        raise ValueError("dpgrf: the adjacency matrix is not square")
    if any(abs(A[i][j] - A[j][i]) > _EPS
           for i in range(n) for j in range(n)):
        raise ValueError("dpgrf: the adjacency matrix must be "
                         "symmetric")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if A[i][j] != 0.0]
    return {"pairs": pairs, "n_pairs": len(pairs), "n_regions": n,
            "degrees": [sum(A[i]) for i in range(n)]}


def car_precision(W, rho=0.99, tau=1.0):
    r""":math:`\tau(D-\rho W)` with :math:`D` the diagonal of row
    sums.

    :math:`\rho = 1` gives the intrinsic CAR, which is singular -- a
    fact the model has to live with rather than hide.
    """
    A = [[float(v) for v in r] for r in k.mat(W)]
    n = len(A)
    r_ = float(rho)
    if not 0.0 <= r_ <= 1.0:
        raise ValueError("dpgrf: rho must lie in [0,1]")
    D = [sum(A[i]) for i in range(n)]
    if any(v <= 0.0 for v in D):
        raise ValueError("dpgrf: a region has no neighbours, so its "
                         "conditional variance is undefined")
    Q = [[float(tau) * ((D[i] if i == j else 0.0) - r_ * A[i][j])
          for j in range(n)] for i in range(n)]
    vals, _ = np.linalg.eigh(Q)
    return {"Q": Q, "degrees": D, "rho": r_,
            "min_eigenvalue": min(vals),
            "singular": min(vals) < 1e-9,
            "note": "rho = 1 is the intrinsic CAR and IS singular; "
                    "it is a proper prior only up to a constant"}


def sample_labels(n_regions, alpha, rng=None, seed=0):
    r"""Cluster labels from the Pólya urn.

    The DP is what puts positive probability on two regions taking
    the SAME value.
    """
    return urn.sample_urn(int(n_regions), float(alpha), rng, seed)


def coclustering(label_draws):
    r"""Posterior :math:`P(z_i = z_j)` from sampled labels.

    Symmetric with a unit diagonal by construction, which is a cheap
    check on the indexing.
    """
    L = [list(d) for d in label_draws]
    if not L:
        raise ValueError("dpgrf: no label draws given")
    n = len(L[0])
    if any(len(d) != n for d in L):
        raise ValueError("dpgrf: the draws differ in length")
    M = [[0.0] * n for _ in range(n)]
    for d in L:
        for i in range(n):
            for j in range(n):
                if d[i] == d[j]:
                    M[i][j] += 1.0
    M = [[v / len(L) for v in row] for row in M]
    return {"matrix": M, "n_draws": len(L), "n": n,
            "symmetric": all(abs(M[i][j] - M[j][i]) < 1e-12
                             for i in range(n) for j in range(n)),
            "unit_diagonal": all(abs(M[i][i] - 1.0) < 1e-12
                                 for i in range(n))}


def boundary_probabilities(W, label_draws, threshold=0.5):
    r""":math:`P(\phi_i\ne\phi_j)` for ADJACENT pairs.

    A probability, not a thresholded difference -- which is the point
    of using a prior that can produce ties.
    """
    pairs = adjacency_pairs(W)["pairs"]
    co = coclustering(label_draws)["matrix"]
    out = []
    for (i, j) in pairs:
        p = 1.0 - co[i][j]
        out.append({"pair": (i, j), "p_difference": p,
                    "boundary": p > float(threshold)})
    out.sort(key=lambda d: -d["p_difference"])
    return RichResult(payload={
        "estimate": [d["pair"] for d in out if d["boundary"]],
        "boundaries": [d["pair"] for d in out if d["boundary"]],
        "ranked": out, "n_adjacent": len(pairs),
        "n_boundaries": sum(1 for d in out if d["boundary"]),
        "threshold": float(threshold),
        "method": "areal difference boundaries by DP clustering; Li, "
                  "Banerjee, Hanson & McBean (2015)",
        "note": "each number is a posterior probability that two "
                "adjacent regions DIFFER, not a rescaled gap",
    })


def continuous_prior_tie_probability():
    r"""Zero. That is the whole problem.

    Under any continuous prior the posterior probability that two
    regions are exactly equal is 0, so a boundary can only be defined
    by thresholding a difference.
    """
    return {"probability": 0.0,
            "note": "a continuous prior assigns no mass to a point, "
                    "so 'are these two equal?' cannot be answered -- "
                    "which is why the DP is used here"}


def cheatsheet():
    return ("dpgrf: areal wombling asks WHERE THE MAP BREAKS. Under a "
            "continuous prior (plain CAR random effects) the "
            "probability that two regions are EQUAL is exactly zero, "
            "so a boundary can only be a threshold on a difference -- "
            "and the threshold is arbitrary. A DIRICHLET PROCESS prior "
            "puts positive mass on ties, so regions genuinely cluster "
            "and P(phi_i != phi_j) for adjacent i, j is an honest "
            "posterior probability. Keep the CAR part too: the DP "
            "handles ties, the CAR handles the map. Co-clustering is "
            "symmetric with a unit diagonal -- a free check on the "
            "indexing.")


# compact alias per ledger/NAMING.md
dp_grouped_random_field = boundary_probabilities
