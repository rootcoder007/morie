# morie.fn -- function file (rootcoder007/morie)
"""Principal Neighbourhood Aggregation (PNA).

Source CONSULTED: Corso, G., Cavalleri, L., Beaini, D., Lio, P. &
Velickovic, P. (2020), "Principal Neighbourhood Aggregation for Graph
Nets", *NeurIPS 33*; arXiv:2004.05718.  The equations below are the
paper's own, read from the arXiv PDF:

    eq. (2)  max_i(X) = max_{j in N(i)} X_j
             min_i(X) = min_{j in N(i)} X_j
    eq. (3)  sigma_i(X) = sqrt( ReLU( mu_i(X^2) - mu_i(X)^2 ) + eps )
    eq. (5)  S_amp(d) = log(d + 1) / delta,
             delta = (1/|train|) sum_{i in train} log(d_i + 1)
    eq. (6)  S(d, alpha) = ( log(d + 1) / delta )^alpha,  -1 <= alpha <= 1
    eq. (7)  the PNA operator is the tensor product of the scaler
             column [ I ; S(D, alpha=1) ; S(D, alpha=-1) ] with the
             aggregator column [ mu ; sigma ; max ; min ]

The mean aggregator mu is the paper's eq. (1).  ``alpha = 0`` reproduces
the identity scaler I, since S(d, 0) = 1.
"""

import math

from ._richresult import RichResult

__all__ = ["pna"]

# The paper adds a small positive eps inside the square root of eq. (3)
# purely so that sigma stays differentiable; the reference
# implementation uses 1e-5.
_EPS = 1e-5

_AGGREGATORS = ("mean", "std", "max", "min")
_SCALERS = ("identity", "amplification", "attenuation")
_ALPHA = {"identity": 0.0, "amplification": 1.0, "attenuation": -1.0}


def _neighbours(A):
    n = len(A)
    out = []
    for i in range(n):
        row = A[i]
        if len(row) != n:
            raise ValueError("A must be a square adjacency matrix")
        out.append([j for j in range(n) if float(row[j]) != 0.0])
    return out


def _aggregate(name, vals):
    """Aggregators of eq. (1)-(3) over one neighbourhood, one feature."""
    m = len(vals)
    if m == 0:
        return 0.0
    if name == "mean":
        return sum(vals) / m
    if name == "max":
        return max(vals)
    if name == "min":
        return min(vals)
    if name == "std":
        mu = sum(vals) / m
        mu2 = sum(v * v for v in vals) / m
        var = mu2 - mu * mu
        if var < 0.0:
            var = 0.0  # the ReLU of eq. (3)
        return math.sqrt(var + _EPS)
    raise ValueError("unknown aggregator %r" % (name,))


def pna(A, X, aggregators=_AGGREGATORS, scalers=_SCALERS):
    """Principal Neighbourhood Aggregation, Corso et al. (2020) eq. (7).

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix.  A non-zero entry A[i][j] makes j a neighbour
        of i.  The degree d_i is the number of such neighbours.
    X : array-like, shape (n, f)
        Node features.
    aggregators : sequence of {'mean', 'std', 'max', 'min'}
        Aggregators of eq. (1)-(3), in output order.
    scalers : sequence of {'identity', 'amplification', 'attenuation'}
        Scalers of eq. (6) with alpha = 0, +1, -1 respectively.

    Returns
    -------
    RichResult
        ``out`` (n x (len(scalers) * len(aggregators) * f), scaler-major
        as in the eq. (7) tensor product), ``aggregated``, ``degree``,
        ``delta``, ``scale``, ``n``, ``n_features``, ``n_columns``.
    """
    Am = [[float(v) for v in row] for row in A]
    n = len(Am)
    if n == 0:
        raise ValueError("A is empty")
    Xm = [[float(v) for v in row] for row in X]
    if len(Xm) != n:
        raise ValueError("X must have one row per node")
    f = len(Xm[0])
    if any(len(r) != f for r in Xm):
        raise ValueError("X must be rectangular")
    aggs = list(aggregators)
    scls = list(scalers)
    for a in aggs:
        if a not in _AGGREGATORS:
            raise ValueError("unknown aggregator %r" % (a,))
    for s in scls:
        if s not in _SCALERS:
            raise ValueError("unknown scaler %r" % (s,))

    nb = _neighbours(Am)
    deg = [len(nb[i]) for i in range(n)]
    # eq. (5): delta is the mean of log(d + 1) over the training nodes;
    # with a single graph in hand every node is a training node.
    delta = sum(math.log(d + 1.0) for d in deg) / n
    if delta <= 0.0:
        raise ValueError("delta is zero: the graph has no edges")

    # aggregated[i][a][c] -- eq. (1)-(3)
    aggregated = []
    for i in range(n):
        per_agg = []
        for a in aggs:
            per_agg.append([_aggregate(a, [Xm[j][c] for j in nb[i]])
                            for c in range(f)])
        aggregated.append(per_agg)

    # eq. (6) scale factors per node per scaler
    scale = []
    for i in range(n):
        base = math.log(deg[i] + 1.0) / delta
        row = []
        for s in scls:
            al = _ALPHA[s]
            if al == 0.0:
                row.append(1.0)
            elif base <= 0.0:
                # d = 0 gives log(1)/delta = 0; eq. (6) is stated for
                # d > 0, so an isolated node contributes nothing under
                # amplification and is left unscaled under attenuation.
                row.append(0.0 if al > 0.0 else 1.0)
            else:
                row.append(base ** al)
        scale.append(row)

    # eq. (7): scalers (outer) tensor aggregators (inner)
    out = []
    for i in range(n):
        row = []
        for si in range(len(scls)):
            for ai in range(len(aggs)):
                for c in range(f):
                    row.append(scale[i][si] * aggregated[i][ai][c])
        out.append(row)

    return RichResult(payload={
        "out": out,
        "aggregated": aggregated,
        "degree": deg,
        "delta": float(delta),
        "scale": scale,
        "aggregators": aggs, "scalers": scls,
        "n": n, "n_features": f, "n_columns": len(scls) * len(aggs) * f,
        "method": "Corso et al. (2020) PNA, eq. (7) of arXiv:2004.05718"})


def cheatsheet():
    return "pnaG: Corso et al. (2020) principal neighbourhood aggregation"


# compact alias per ledger/NAMING.md
pnaagg = pna
