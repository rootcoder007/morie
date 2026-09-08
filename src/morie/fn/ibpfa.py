# morie.fn -- function file (rootcoder007/morie)
r"""The Indian buffet process: latent features without fixing how many.

A mixture model assigns each object to one class. Many objects are
better described by *several* features at once, and the question then
becomes how many features exist. The IBP answers it by not answering:
a distribution over binary matrices with a finite number of rows and
an **unbounded** number of columns, from which the number of features
actually used is inferred.

**The metaphor is the definition.** Customers enter a buffet with
infinitely many dishes. Customer :math:`i` takes each already-tasted
dish with probability :math:`m_k/i` -- the number who took it before,
over the number of customers so far -- and then takes
:math:`\mathrm{Poisson}(\alpha/i)` *new* dishes. Popularity is
self-reinforcing, and the flow of genuinely new dishes decays as
:math:`1/i`.

**Two consequences worth stating separately.** The expected total
number of dishes is :math:`\alpha H_n \approx \alpha\log n`, so
features grow with the data but only logarithmically. And the expected
number of dishes *per customer* is :math:`\alpha` -- constant, however
large the dataset. ``expected_features`` computes both, because
confusing them is the usual error: the matrix is sparse in a specific
way, not merely large.

**Exchangeability, and the reason it matters.** The distribution over
**left-ordered** binary matrices is exchangeable in the customers, so
the order of arrival does not affect the answer. That is what makes
Gibbs sampling valid: any object can be treated as the last one to
arrive, and resampled with the others held fixed. The anchor checks
the exchangeability rather than assuming it.

References
----------
Griffiths, T. L. & Ghahramani, Z. (2011) "The Indian Buffet Process:
An Introduction and Review", *Journal of Machine Learning Research*
12, 1185-1224. The buffet metaphor and the sequential construction
with probability m_k/i for existing dishes and Poisson(alpha/i) new
dishes; the equivalence class of left-ordered binary matrices and the
exchangeability of the resulting distribution; the expected number of
non-zero entries and of features; and the use of the IBP as a prior
for infinite latent feature models with Gibbs sampling.

Griffiths, T. L. & Ghahramani, Z. (2006) "Infinite Latent Feature
Models and the Indian Buffet Process", *NIPS 2005*, 475-482. The
original.

Teh, Y. W., Gorur, D. & Ghahramani, Z. (2007) "Stick-breaking
Construction for the Indian Buffet Process", *AISTATS 2007*, PMLR 2,
556-563. The stick-breaking representation.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sample_ibp", "expected_features", "left_ordered_form",
           "ibp_log_probability", "gibbs_feature_update"]

_EPS = 1e-12


def sample_ibp(n, alpha, seed=0):
    r"""Draw a binary feature matrix from the buffet.

    Customer :math:`i` takes dish :math:`k` with probability
    :math:`m_k/i` and then :math:`\mathrm{Poisson}(\alpha/i)` new
    ones.
    """
    N = int(n)
    a = float(alpha)
    if N < 1 or a <= 0.0:
        raise ValueError("ibpfa: need n >= 1 and alpha > 0")
    rng = np.random.default_rng(seed)
    rows, counts = [], []
    for i in range(1, N + 1):
        row = []
        for kk in range(len(counts)):
            p = counts[kk] / float(i)
            take = 1 if float(rng.uniform()) < p else 0
            row.append(take)
            counts[kk] += take
        lam = a / float(i)
        new, L, p = 0, math.exp(-lam), float(rng.uniform())
        cum = L
        while p > cum and new < 100:
            new += 1
            L *= lam / new
            cum += L
        for _ in range(new):
            row.append(1)
            counts.append(1)
        rows.append(row)
    K = len(counts)
    Z = [r + [0] * (K - len(r)) for r in rows]
    return {"Z": Z, "K": K, "counts": counts, "alpha": a, "n": N,
            "features_per_object": [sum(r) for r in Z],
            "note": "the number of features is INFERRED, not fixed"}


def expected_features(n, alpha):
    r"""Total features :math:`\alpha H_n` against per-object
    :math:`\alpha`.

    Two different quantities: the matrix grows logarithmically in
    width while each row's weight stays constant.
    """
    N = int(n)
    a = float(alpha)
    if N < 1 or a <= 0.0:
        raise ValueError("ibpfa: need n >= 1 and alpha > 0")
    H = sum(1.0 / i for i in range(1, N + 1))
    return {"expected_total_features": a * H,
            "harmonic": H,
            "expected_per_object": a,
            "expected_nonzeros": a * N,
            "note": "total grows like alpha log n; per object it is "
                    "CONSTANT at alpha"}


def left_ordered_form(Z):
    r"""The canonical left-ordered matrix.

    Columns are an unordered set; ordering them by their binary
    history is what defines the equivalence class the distribution
    lives on.
    """
    M = [[int(v) for v in r] for r in k.mat(Z)]
    if not M:
        raise ValueError("ibpfa: the matrix is empty")
    n, K = len(M), len(M[0])
    hist = []
    for kk in range(K):
        h = 0
        for i in range(n):
            h = (h << 1) | M[i][kk]
        hist.append((h, kk))
    hist.sort(key=lambda t: (-t[0], t[1]))
    order = [kk for _, kk in hist]
    return {"Z": [[M[i][kk] for kk in order] for i in range(n)],
            "order": order,
            "note": "columns are an unordered SET; left-ordering "
                    "picks the canonical representative"}


def ibp_log_probability(Z, alpha):
    r"""Log probability of the left-ordered matrix.

    Depends on the column *counts* only, which is exactly why the
    distribution is exchangeable in the customers.
    """
    M = [[int(v) for v in r] for r in k.mat(Z)]
    n, K = len(M), len(M[0]) if M else 0
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("ibpfa: alpha must be positive")
    H = sum(1.0 / i for i in range(1, n + 1))
    lp = K * math.log(a) - a * H
    for kk in range(K):
        m = sum(M[i][kk] for i in range(n))
        if m == 0:
            continue
        lp += (k.lgamma(n - m + 1) + k.lgamma(m)
               - k.lgamma(n + 1))
    return lp


def gibbs_feature_update(Z, i, kk, likelihood, alpha):
    r"""Resample one entry, treating object :math:`i` as the last to
    arrive.

    Valid because of exchangeability: any customer may be considered
    last, so the conditional is
    :math:`P(z_{ik}=1 \mid z_{-i,k}) = m_{-i,k}/n`.
    """
    M = [[int(v) for v in r] for r in k.mat(Z)]
    n = len(M)
    m_minus = sum(M[j][kk] for j in range(n) if j != i)
    if m_minus == 0:
        return {"z": 0, "prior": 0.0,
                "note": "a feature held by nobody else is dropped; "
                        "new ones arrive through the Poisson draw"}
    prior = m_minus / float(n)
    on, off = [list(r) for r in M], [list(r) for r in M]
    on[i][kk], off[i][kk] = 1, 0
    l1 = float(likelihood(on)) + math.log(max(prior, _EPS))
    l0 = float(likelihood(off)) + math.log(max(1.0 - prior, _EPS))
    mx = max(l1, l0)
    p1 = math.exp(l1 - mx) / (math.exp(l1 - mx) + math.exp(l0 - mx))
    return {"p_on": p1, "prior": prior,
            "z": 1 if p1 > 0.5 else 0,
            "note": "exchangeability is what licenses treating i as "
                    "the last customer"}


def cheatsheet():
    return ("ibpfa: objects have SEVERAL latent features, and how many "
            "exist is unknown -- so use a distribution over binary "
            "matrices with unboundedly many columns. Customer i takes "
            "an existing dish with probability m_k/i (popularity "
            "self-reinforces) and Poisson(alpha/i) NEW dishes (the "
            "flow decays as 1/i). Two different numbers: expected "
            "TOTAL features alpha*H_n ~ alpha log n, expected features "
            "PER OBJECT constant at alpha. The left-ordered form is "
            "EXCHANGEABLE, which is what licenses Gibbs sampling by "
            "treating any object as the last to arrive.")


# compact alias per ledger/NAMING.md
indianbuffet = sample_ibp

# public names resolved by fn/_lazy_map.json
indian_buffet_factor = sample_ibp
