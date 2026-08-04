# morie.fn -- function file (rootcoder007/morie)
"""The sparse vector technique (differential privacy)."""

from math import fsum

from ._richresult import RichResult
from ._spx import vec

__all__ = [
    "sparse_vector",
    "sparsevector",
]


def sparse_vector(queries, threshold, c=1, epsilon=1.0,
                  threshold_noise=0.0, query_noise=None):
    """AboveThreshold / sparse vector, with the noise supplied by the caller.

    NOT IN SCHABENBERGER & GOTWAY -- this is differential privacy, not
    spatial statistics; the module only shares the ``sp`` prefix. The
    algorithm is Dwork, C. & Roth, A. (2014), *The Algorithmic Foundations
    of Differential Privacy*, Alg. 2 (AboveThreshold) and its c-fold
    extension (Sparse Vector), building on Hardt, M. & Rothblum, G. (2010),
    "A multiplicative weights mechanism for privacy-preserving data
    analysis", FOCS. Named from the general literature; NOT verified
    against a PDF in this corpus.

    The point of the mechanism is that the privacy cost scales with c, the
    number of ABOVE-threshold answers released, and not with the number of
    queries asked. A stream of a million below-threshold queries costs the
    same as a stream of ten, which is why it is the workhorse for
    "find me the few interesting cells".

    Budget split, as in Dwork-Roth: eps/2 perturbs the threshold once,
    eps/2 is divided over the c releases, so each above-threshold answer
    is perturbed at eps/(2c).

    THE NOISE IS AN ARGUMENT, NOT DRAWN HERE. A function that samples its
    own Laplace noise cannot be compared across language arms, because R's
    and Python's generators do not produce the same stream from the same
    seed. Pass `threshold_noise` and `query_noise` explicitly; the
    defaults are zero, which gives the NON-PRIVATE decision sequence and
    is useful only for testing the control flow. The returned
    ``noise_scales`` are the Laplace scales a caller must use for the
    stated epsilon, assuming sensitivity 1.

    Parameters
    ----------
    queries : (m,) array-like
        Query answers, in the order asked.
    threshold : float
        The threshold T.
    c : int
        Number of above-threshold answers to release before halting.
    epsilon : float
        Total privacy budget; must be positive.
    threshold_noise : float
        The single perturbation added to T.
    query_noise : (m,) array-like, optional
        Per-query perturbations; defaults to zeros.

    Returns
    -------
    RichResult
        ``above``, ``released``, ``halted_at``, ``n_above``,
        ``noisy_threshold``, ``noise_scales``, ``epsilon_split``, ``n``,
        ``method``.
    """
    q = vec(queries, "queries")
    m = len(q)
    t = float(threshold)
    c = int(c)
    eps = float(epsilon)
    if c < 1:
        raise ValueError("`c` must be at least 1")
    if c > m:
        raise ValueError("`c` (%d) exceeds the number of queries (%d)"
                         % (c, m))
    if eps <= 0:
        raise ValueError("`epsilon` must be positive")
    if query_noise is None:
        qn = [0.0] * m
    else:
        qn = vec(query_noise, "query_noise")
        if len(qn) != m:
            raise ValueError("`query_noise` must have one entry per query")

    tn = t + float(threshold_noise)
    above = []
    released = []
    hits = 0
    halted = m
    for i in range(m):
        if hits >= c:
            halted = i
            break
        if q[i] + qn[i] >= tn:
            above.append(True)
            released.append(q[i] + qn[i])
            hits = hits + 1
        else:
            above.append(False)
            released.append(None)
    while len(above) < m:
        above.append(None)
        released.append(None)

    return RichResult(payload={
        "above": above,
        "released": released,
        "halted_at": float(halted),
        "n_above": float(hits),
        "noisy_threshold": tn,
        "noise_scales": {"threshold": 2.0 / eps, "query": 2.0 * c / eps},
        "epsilon_split": {"threshold": eps / 2.0, "queries": eps / 2.0},
        "epsilon": eps,
        "c": float(c),
        "cost_scales_with_c_not_with_m": True,
        "answered": fsum([1.0 for a in above if a is not None]),
        "n": m,
        "method": ("Sparse vector / AboveThreshold (Dwork & Roth 2014, "
                   "Alg. 2; Hardt & Rothblum 2010) with caller-supplied "
                   "noise; NOT in Schabenberger & Gotway"),
    })


def cheatsheet():
    return "sparsv: sparse vector technique with supplied noise"


# compact alias per ledger/NAMING.md
sparsevector = sparse_vector
