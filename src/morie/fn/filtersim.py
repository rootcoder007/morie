# morie.fn -- function file (rootcoder007/morie)
"""FILTERSIM multiple-point simulation: training-image patterns summarised by six directional filter scores,
classified into prototypes by k-means, and pasted along a random path by closest-prototype matching."""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["filtersim"]


class _U:
    def __init__(self, seed):
        self.seed, self.block, self.buf, self.pos = seed, 0, [], 0

    def __call__(self):
        if self.pos >= len(self.buf):
            self.buf = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def _filters(m):
    """Six directional filters over offsets -m..m (Zhang, Switzer and Journel 2006): average, gradient, curvature in x and y."""
    offs = [(du, dv) for dv in range(-m, m + 1) for du in range(-m, m + 1)]
    f = []
    for axis in (0, 1):
        for kind in ("avg", "grad", "curv"):
            row = []
            for o in offs:
                u = o[axis]
                if kind == "avg":
                    row.append(1.0 - abs(u) / m)
                elif kind == "grad":
                    row.append(u / m)
                else:
                    row.append(2.0 * abs(u) / m - 1.0)
            f.append(row)
    return offs, f


def _lsum(v):
    s = 0.0
    for a in v:
        s += a
    return s


def filtersim(
    training_image,
    nx: int,
    ny: int,
    *,
    template: int = 7,
    inner: int | None = None,
    n_classes: int = 10,
    hard_data=None,
    seed: int = 1,
    max_kmeans: int = 50,
) -> RichResult:
    r"""FILTERSIM simulation of an ``ny x nx`` grid from a 2-D training image (Zhang, Switzer and Journel 2006).

    Every full ``template x template`` window of ``training_image``
    (``ti[y][x]``) is a pattern; six filters (average ``1 - |u|/m``,
    gradient ``u/m`` and curvature ``2|u|/m - 1`` along x and along y,
    ``m = template // 2``) reduce it to six scores. Standardised scores
    are clustered by k-means (k-means++ seeding, Lloyd iterations) into
    ``n_classes`` classes whose mean patterns are the prototypes. Along a
    random path over uninformed nodes, the data event in the template
    (hard data and nodes already pasted) is compared with every prototype
    by mean squared difference over the informed offsets; a training
    pattern drawn uniformly from the closest class (a random class,
    weighted by size, when nothing is informed) has its central
    ``(2 inner + 1)^2`` patch pasted onto the still-uninformed nodes.
    ``hard_data`` maps ``(x, y)`` to values that are honoured. Philox
    draws, so the R twin gives the same realisation.

    References
    ----------
    Zhang, T., Switzer, P. and Journel, A. (2006). Filter-based
    classification of training image patterns for spatial simulation.
    *Mathematical Geology*, 38, 63-80.
    Mariethoz, G. and Caers, J. (2014). *Multiple-point Geostatistics*.
    Wiley-Blackwell.

    Examples
    --------
    >>> ti = [[1.0 if (x // 3) % 2 == 0 else 0.0 for x in range(24)] for y in range(24)]
    >>> r = filtersim(ti, 12, 6, template=5, n_classes=4, seed=2)
    >>> [int(v) for v in r.realisation[0]]
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    """
    ti = [[float(v) for v in row] for row in training_image]
    H, W = len(ti), len(ti[0])
    if template % 2 == 0 or template < 3:
        raise ValueError("template must be an odd integer >= 3")
    m = template // 2
    ip = m // 2 if inner is None else int(inner)
    if template > H or template > W:
        raise ValueError("training image smaller than the template")
    offs, F = _filters(m)
    pats, scores = [], []
    for y in range(m, H - m):
        for x in range(m, W - m):
            p = [ti[y + dv][x + du] for du, dv in offs]
            pats.append(p)
            scores.append([_lsum(f[k] * p[k] for k in range(len(p))) for f in F])
    n = len(pats)
    K = max(1, min(int(n_classes), n))
    mu = [_lsum(s[j] for s in scores) / n for j in range(6)]
    sd = [math.sqrt(_lsum((s[j] - mu[j]) ** 2 for s in scores) / n) or 1.0 for j in range(6)]
    Z = [[(s[j] - mu[j]) / sd[j] for j in range(6)] for s in scores]
    U = _U(seed)

    def d2(a, b):
        s = 0.0
        for j in range(6):
            s += (a[j] - b[j]) ** 2
        return s

    cent = [list(Z[min(int(U() * n), n - 1)])]
    while len(cent) < K:
        dist = [min(d2(z, c) for c in cent) for z in Z]
        tot = _lsum(dist)
        if tot == 0:
            break
        r, acc, pick = U() * tot, 0.0, n - 1
        for i, dv in enumerate(dist):
            acc += dv
            if r < acc:
                pick = i
                break
        cent.append(list(Z[pick]))
    K = len(cent)
    lab = [0] * n
    for itr in range(int(max_kmeans)):
        new = [min(range(K), key=lambda c, z=z: (d2(z, cent[c]), c)) for z in Z]
        for c in range(K):
            mem = [Z[i] for i in range(n) if new[i] == c]
            if mem:
                cent[c] = [_lsum(z[j] for z in mem) / len(mem) for j in range(6)]
        if new == lab and itr > 0:
            break
        lab = new
    members = [[i for i in range(n) if lab[i] == c] for c in range(K)]
    proto = []
    for c in range(K):
        if members[c]:
            proto.append([_lsum(pats[i][k] for i in members[c]) / len(members[c]) for k in range(len(offs))])
        else:
            proto.append(None)
    grid = [[None] * nx for _ in range(ny)]
    hard = {}
    for (hx, hy), v in dict(hard_data or {}).items():
        grid[int(hy)][int(hx)] = float(v)
        hard[(int(hx), int(hy))] = True
    nodes = [(x, y) for y in range(ny) for x in range(nx)]
    for k in range(len(nodes) - 1, 0, -1):
        j = min(int(U() * (k + 1)), k)
        nodes[k], nodes[j] = nodes[j], nodes[k]
    used_classes = []
    for x, y in nodes:
        if grid[y][x] is not None:
            continue
        ev = [
            (k, grid[y + dv][x + du])
            for k, (du, dv) in enumerate(offs)
            if 0 <= x + du < nx and 0 <= y + dv < ny and grid[y + dv][x + du] is not None
        ]
        if not ev:
            r, acc, cls = U() * n, 0.0, K - 1
            for c in range(K):
                acc += len(members[c])
                if r < acc:
                    cls = c
                    break
        else:
            best, cls = math.inf, 0
            for c in range(K):
                if proto[c] is None:
                    continue
                s = 0.0
                for k, v in ev:
                    s += (proto[c][k] - v) ** 2
                s /= len(ev)
                if s < best:
                    best, cls = s, c
        mem = members[cls]
        pi = mem[min(int(U() * len(mem)), len(mem) - 1)]
        used_classes.append(cls)
        for k, (du, dv) in enumerate(offs):
            if abs(du) <= ip and abs(dv) <= ip:
                xx, yy = x + du, y + dv
                if 0 <= xx < nx and 0 <= yy < ny and grid[yy][xx] is None:
                    grid[yy][xx] = pats[pi][k]
    return RichResult(
        payload={
            "realisation": grid,
            "n_patterns": n,
            "class_sizes": [len(mm) for mm in members],
            "prototypes": proto,
            "classes_used": used_classes,
        }
    )


def cheatsheet() -> str:
    return "filtersim(ti, nx, ny) -> FILTERSIM realisation (filter scores, k-means prototypes, patch pasting)."
