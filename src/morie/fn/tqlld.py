r"""Lloyd-Max optimal scalar quantiser.

Lloyd, S. P. (1982) "Least squares quantization in PCM", *IEEE
Transactions on Information Theory* **28**(2), 129-137 (written 1957);
Max, J. (1960) "Quantizing for minimum distortion", *IRE Transactions on
Information Theory* **6**(1), 7-12.

For a source with density :math:`p`, the quantiser minimising mean
squared error satisfies two conditions simultaneously, and the algorithm
is just alternating between them until they hold:

**Nearest-neighbour condition.** Each decision boundary sits midway
between its neighbouring codewords,

.. math:: b_k = \frac{y_k + y_{k+1}}{2}.

**Centroid condition.** Each codeword is the conditional mean of its
cell,

.. math:: y_k = \frac{\int_{b_{k-1}}^{b_k} x\,p(x)\,dx}
                     {\int_{b_{k-1}}^{b_k} p(x)\,dx}
          = E[X \mid X \in \text{cell } k].

Distortion is non-increasing at every half-step -- each condition is
optimal given the other -- so the iteration converges monotonically.
That monotonicity is the property worth testing, because it is what
distinguishes this from any other clustering-shaped loop.

Routes
------
``source``:

``"gaussian"``
    The N(0,1) codebook by numerical integration on a fine grid. Max
    (1960) Table I is the classical reference for these values.
``"empirical"``
    Lloyd's algorithm on supplied samples -- 1-D k-means, where the
    centroid step is the plain mean of the cell.
``"uniform"``
    Closed form on :math:`[a, b]`: cells of equal width
    :math:`(b-a)/N` with codewords at their midpoints, which is exactly
    what the two conditions give for a flat density.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["lloyd_max_codebook", "quantize_with_codebook", "tqlld"]

_SOURCES = ("gaussian", "empirical", "uniform")


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gaussian_cells(bounds, lo, hi, n_grid):
    """Mass and first moment of N(0,1) on each cell, by fine quadrature."""
    edges = [lo] + list(bounds) + [hi]
    mass, mom = [], []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            mass.append(0.0)
            mom.append(0.0)
            continue
        m = max(2, int(n_grid * (b - a) / (hi - lo)) + 2)
        h = (b - a) / m
        s0 = s1 = 0.0
        for i in range(m):
            x = a + (i + 0.5) * h          # midpoint rule
            w = _phi(x) * h
            s0 += w
            s1 += x * w
        mass.append(s0)
        mom.append(s1)
    return mass, mom


def lloyd_max_codebook(levels=4, source="gaussian", data=None, lo=None,
                       hi=None, max_iter=200, tol=1e-12, n_grid=20000):
    r"""Optimal ``levels``-point scalar codebook.

    Returns
    -------
    RichResult
        ``codebook`` are the codewords, ``boundaries`` the decision
        thresholds, ``distortion`` the resulting MSE, and
        ``distortion_history`` shows the monotone decrease.
    """
    N = int(levels)
    if N < 1:
        raise ValueError("lloyd_max_codebook: levels must be >= 1, got %r"
                         % (levels,))
    src = str(source).lower()
    if src not in _SOURCES:
        raise ValueError(
            "lloyd_max_codebook: source must be one of %s, got %r"
            % (", ".join(_SOURCES), source))

    if src == "uniform":
        a = -1.0 if lo is None else float(lo)
        b = 1.0 if hi is None else float(hi)
        if b <= a:
            raise ValueError("lloyd_max_codebook: need hi > lo")
        w = (b - a) / N
        cb = [a + (k + 0.5) * w for k in range(N)]
        bnd = [a + (k + 1) * w for k in range(N - 1)]
        # For a flat density the MSE of a cell of width w is w^2/12.
        dist = w * w / 12.0
        return RichResult(payload={
            "estimate": cb, "codebook": cb, "boundaries": bnd,
            "distortion": dist, "distortion_history": [dist],
            "iterations": 0, "converged": True, "source": src,
            "levels": N, "lo": a, "hi": b,
            "method": "Uniform-source Lloyd-Max, closed form "
                      "(Lloyd 1982; Max 1960)",
        })

    if src == "empirical":
        xs = sorted(float(v) for v in
                    np.atleast_1d(np.asarray(data, dtype=float)))
        if not xs:
            raise ValueError("lloyd_max_codebook: empirical source needs data")
        if len(xs) < N:
            raise ValueError(
                "lloyd_max_codebook: %d samples cannot support %d levels"
                % (len(xs), N))
        # Initial codewords at evenly spaced sample quantiles.
        cb = [xs[min(len(xs) - 1, int((k + 0.5) * len(xs) / N))]
              for k in range(N)]
    else:
        LO, HI = -8.0, 8.0
        # Initial codewords evenly spaced over the bulk of the density.
        cb = [-3.0 + 6.0 * (k + 0.5) / N for k in range(N)]

    hist = []
    it = 0
    converged = False
    prev = float("inf")
    for it in range(1, int(max_iter) + 1):
        cb = sorted(cb)
        # Nearest-neighbour condition: boundaries midway between codewords.
        bnd = [0.5 * (cb[k] + cb[k + 1]) for k in range(N - 1)]

        if src == "empirical":
            cells = [[] for _ in range(N)]
            j = 0
            for x in xs:
                while j < N - 1 and x > bnd[j]:
                    j += 1
                cells[j].append(x)
            new = []
            for k in range(N):
                # Centroid condition; an empty cell keeps its codeword
                # rather than collapsing to NaN.
                new.append(sum(cells[k]) / len(cells[k]) if cells[k] else cb[k])
            dist = sum((x - new[k]) ** 2 for k in range(N) for x in cells[k])
            dist /= float(len(xs))
        else:
            mass, mom = _gaussian_cells(bnd, LO, HI, n_grid)
            new = [(mom[k] / mass[k]) if mass[k] > 1e-300 else cb[k]
                   for k in range(N)]
            # E[(X - y_k)^2] over each cell, using E[X^2] = 1 overall.
            dist = 1.0 - sum(mom[k] * new[k] for k in range(N))

        hist.append(dist)
        shift = max(abs(new[k] - cb[k]) for k in range(N))
        cb = new
        if shift <= tol or abs(prev - dist) <= tol:
            converged = True
            break
        prev = dist

    cb = sorted(cb)
    bnd = [0.5 * (cb[k] + cb[k + 1]) for k in range(N - 1)]
    return RichResult(payload={
        "estimate": cb,
        "codebook": cb,
        "boundaries": bnd,
        "distortion": float(hist[-1]) if hist else 0.0,
        "distortion_history": hist,
        "iterations": int(it),
        "converged": bool(converged),
        "source": src,
        "levels": N,
        "method": "Lloyd-Max alternating nearest-neighbour and centroid "
                  "conditions (Lloyd 1982; Max 1960)",
    })


def quantize_with_codebook(x, codebook):
    """Map each sample to its nearest codeword; returns indices and values."""
    cb = [float(v) for v in codebook]
    if not cb:
        raise ValueError("quantize_with_codebook: codebook is empty")
    idx, val = [], []
    for v in np.atleast_1d(np.asarray(x, dtype=float)):
        v = float(v)
        best = 0
        bd = abs(v - cb[0])
        for k in range(1, len(cb)):
            d = abs(v - cb[k])
            if d < bd:
                bd, best = d, k
        idx.append(best)
        val.append(cb[best])
    mse = (sum((float(a) - b) ** 2 for a, b in
               zip(np.atleast_1d(np.asarray(x, dtype=float)), val))
           / float(len(val)))
    return RichResult(payload={
        "estimate": val, "indices": idx, "values": val, "mse": float(mse),
        "levels": len(cb),
        "method": "Nearest-codeword quantisation",
    })


def cheatsheet():
    return ("tqlld: Lloyd-Max, boundaries b_k = (y_k + y_k+1)/2 and "
            "codewords y_k = E[X | cell k]; distortion is monotone "
            "non-increasing; sources gaussian / empirical / uniform.")


tqlld = lloyd_max_codebook
