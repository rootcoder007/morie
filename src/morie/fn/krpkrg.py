# morie.fn -- function file (rootcoder007/morie)
r"""Ordinary kriging: the BLUP for a spatial field with unknown mean.

Inverse-distance schemes weight by geometry alone. Kriging weights by the
COVARIANCE STRUCTURE the data themselves exhibit, which is why it is the
best linear unbiased predictor and why it returns a variance: the same
configuration of points can be highly informative or nearly useless
depending on the range of the variogram, and only the second is visible
in the prediction error.

Ordinary kriging carries an unbiasedness constraint (weights sum to one)
via a Lagrange multiplier, so the system solved is

.. math:: \begin{pmatrix} \Gamma & 1 \\ 1' & 0\end{pmatrix}
          \begin{pmatrix}\lambda \\ \mu\end{pmatrix}
          = \begin{pmatrix}\gamma_0 \\ 1\end{pmatrix},

with :math:`\Gamma` the matrix of semivariances between data points.
The kriging variance is :math:`\lambda'\gamma_0 + \mu` -- it depends on
the configuration and the model, never on the observed values, which is
exactly the property that makes it a design criterion.

References
----------
Goovaerts, P. (2005) *Geostatistics for Natural Resources Evaluation*,
Oxford University Press, Ch. 5 (ordinary kriging, the unbiasedness
constraint and the kriging variance).

Cressie, N. (1993) *Statistics for Spatial Data*, revised ed., Wiley,
Sec. 3.2 (kriging as best linear unbiased prediction).

Matheron, G. (1963) "Principles of geostatistics", *Economic Geology*
**58**(8), 1246-1266, doi:10.2113/gsecongeo.58.8.1246.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["ordinary_kriging"]

_EPS = 1e-12


def _gamma(h, model, nugget, sill, rng):
    """Semivariogram: nugget + partial sill * structure(h)."""
    if h <= 0.0:
        return 0.0
    ps = sill - nugget
    if rng <= _EPS:
        return sill
    if model == "spherical":
        if h >= rng:
            return sill
        r = h / rng
        return nugget + ps * (1.5 * r - 0.5 * r ** 3)
    if model == "exponential":
        return nugget + ps * (1.0 - math.exp(-3.0 * h / rng))
    if model == "gaussian":
        return nugget + ps * (1.0 - math.exp(-3.0 * (h / rng) ** 2))
    raise ValueError("krpkrg: model must be spherical, exponential or "
                     "gaussian, got %r" % (model,))


def ordinary_kriging(coords, values, targets, model="spherical",
                     nugget=0.0, sill=1.0, rng=1.0):
    r"""Predict at ``targets`` from data at ``coords``."""
    C = [[float(v) for v in r] for r in k.mat(coords)]
    z = [float(v) for v in k.vec(values)]
    Tg = [[float(v) for v in r] for r in k.mat(targets)]
    n = len(C)
    if n == 0:
        raise ValueError("krpkrg: no data locations")
    if len(z) != n:
        raise ValueError("krpkrg: %d locations but %d values" % (n, len(z)))
    if sill < nugget:
        raise ValueError("krpkrg: the sill cannot be below the nugget")
    d = len(C[0])

    def dist(a, b):
        return math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(d)))

    G = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            G[i][j] = _gamma(dist(C[i], C[j]), model, nugget, sill, rng)
        G[i][n] = 1.0
        G[n][i] = 1.0
    G[n][n] = 0.0

    preds, variances, weightsets = [], [], []
    for t in Tg:
        g0 = [_gamma(dist(C[i], t), model, nugget, sill, rng)
              for i in range(n)] + [1.0]
        sol = k.cholsolve([[sum(G[u][a] * G[u][b] for u in range(n + 1))
                            for b in range(n + 1)] for a in range(n + 1)],
                          [sum(G[u][a] * g0[u] for u in range(n + 1))
                           for a in range(n + 1)])
        lam = sol[:n]
        mu = sol[n]
        preds.append(sum(lam[i] * z[i] for i in range(n)))
        v = sum(lam[i] * g0[i] for i in range(n)) + mu
        # exactly 0 at a data location; below the floor it is rounding, and
        # sqrt would turn that rounding into a spurious standard error
        floor = 1e-12 * (sill if sill > _EPS else 1.0)
        variances.append(0.0 if v < floor else v)
        weightsets.append(lam)

    return RichResult(payload={
        "estimate": preds, "prediction": preds, "variance": variances,
        "std_error": [math.sqrt(v) for v in variances],
        "weights": weightsets, "n": n, "n_targets": len(Tg),
        "model": model, "nugget": float(nugget), "sill": float(sill),
        "range": float(rng),
        "method": "ordinary kriging with a Lagrange unbiasedness constraint "
                  "(Goovaerts 2005 Ch. 5)",
        "note": "the kriging variance depends on the configuration and the "
                "variogram, never on the observed values -- which is what "
                "makes it usable as a design criterion",
    })


def cheatsheet():
    return ("krpkrg: ordinary_kriging(coords, values, targets, model, "
            "nugget, sill, range) -> BLUP and kriging variance "
            "(Goovaerts 2005, Geostatistics for Natural Resources "
            "Evaluation, Ch. 5)")
