# morie.fn -- function file (rootcoder007/morie)
r"""Convergent cross mapping: causality from a single time series pair.

**The problem with correlation, and with Granger.** In a coupled
deterministic system, two variables can be strongly correlated
without either driving the other, or driven by each other with no
correlation at all. Granger causality asks whether the past of
:math:`Y` improves prediction of :math:`X`, which assumes the
information in :math:`Y` is *separable* from that in :math:`X`. In a
deterministic dynamical system it is not: by Takens' theorem the
delay embedding of either variable reconstructs the whole attractor.

**Turning that into a test.** If :math:`X` influences :math:`Y`, then
the state of :math:`X` leaves a signature in :math:`Y`'s trajectory,
so :math:`Y`'s reconstructed manifold :math:`M_Y` can be used to
*recover* :math:`X`. Nearest neighbours on :math:`M_Y` identify times
at which the system was in a similar state; the corresponding values
of :math:`X` are averaged with exponentially decaying weights,

.. math:: w_i \propto \exp(-d_i / d_1), \qquad
          \hat{X}(t) \mid M_Y = \sum_{i=1}^{E+1} w_i X(t_i),

and the skill :math:`\rho` is the correlation between :math:`X` and
:math:`\hat{X}`.

**The direction is the counterintuitive part.** High skill for
"*Y* cross maps *X*" means **X causes Y**. The effect carries the
signature of the cause, not the other way round. Getting this
backwards is the standard way to misread a CCM plot, so both
directions are always computed and each is labelled with the causal
statement it supports.

**Convergence is the test, not the level.** A high :math:`\rho` at
one library length proves nothing -- correlated noise gives that. The
claim is that :math:`\rho` *rises with library length* and saturates,
because a longer trajectory fills in the attractor and nearest
neighbours get genuinely nearer. Both the curve and the increase
from shortest to longest library are reported; reading only the final
:math:`\rho` throws the test away.

**What it cannot do.** Strongly synchronised variables give high skill
in both directions and the method cannot separate them; purely
stochastic series have no attractor to reconstruct. Neither case is
silently reported as causality -- the convergence figures make it
visible.

References
----------
Sugihara, G., May, R., Ye, H., Hsieh, C.-h., Deyle, E., Fogarty, M. &
Munch, S. (2012) "Detecting causality in complex ecosystems",
*Science* 338(6106), 496-500, doi:10.1126/science.1227079. The
cross-map construction above, the exponential neighbour weighting,
the convergence criterion, the direction convention (Y cross mapping
X indicates X drives Y), and the coupled logistic system used as the
worked demonstration.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["embed", "cross_map", "ccm", "coupled_logistic",
           "convergent_cross_mapping"]


def embed(series, E=2, tau=1):
    r"""Time-delay embedding: vectors :math:`(x_t, x_{t-\tau},
    \ldots)` with their time indices."""
    v = [float(x) for x in series]
    e, t = int(E), int(tau)
    if e < 1:
        raise ValueError("cnvlfc: the embedding dimension must be at "
                         "least 1")
    if t < 1:
        raise ValueError("cnvlfc: the delay must be at least 1")
    need = (e - 1) * t
    if len(v) <= need + 1:
        raise ValueError("cnvlfc: %d points cannot support dimension "
                         "%d at delay %d" % (len(v), e, t))
    idx = list(range(need, len(v)))
    pts = [[v[i - k * t] for k in range(e)] for i in idx]
    return {"points": pts, "index": idx, "E": e, "tau": t}


def _corr(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((x - mb) ** 2 for x in b)
    if sa <= 0 or sb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb)
               for i in range(n)) / math.sqrt(sa * sb)


def cross_map(driver, response, E=2, tau=1, library=None, seed=1,
              exclude=0):
    r"""Estimate ``driver`` from the manifold of ``response``.

    Skill here supports "``driver`` causes ``response``". ``library``
    limits how much of the trajectory the neighbours may come from;
    ``exclude`` is a Theiler window suppressing temporally adjacent
    neighbours.
    """
    X = [float(v) for v in driver]
    Y = [float(v) for v in response]
    if len(X) != len(Y):
        raise ValueError("cnvlfc: the two series have %d and %d "
                         "points" % (len(X), len(Y)))
    em = embed(Y, E, tau)
    pts, idx = em["points"], em["index"]
    m = len(pts)
    k = int(E) + 1
    if library is None:
        lib = list(range(m))
    else:
        L = int(library)
        if L < k + 1:
            raise ValueError("cnvlfc: a library of %d points cannot "
                             "supply %d neighbours" % (L, k))
        if L > m:
            raise ValueError("cnvlfc: the library asks for %d points "
                             "but only %d are embeddable" % (L, m))
        rng = np.random.default_rng(int(seed))
        start = int(rng.random() * (m - L + 1)) % (m - L + 1)
        lib = list(range(start, start + L))
    libset = set(lib)
    obs, pred = [], []
    for a in range(m):
        cand = [b for b in lib
                if b != a and abs(idx[b] - idx[a]) > int(exclude)]
        if len(cand) < k:
            continue
        d = sorted(((math.sqrt(sum((pts[a][c] - pts[b][c]) ** 2
                                   for c in range(len(pts[a])))), b)
                    for b in cand))[:k]
        d1 = d[0][0]
        if d1 <= 0:
            w = [1.0 if j == 0 else 0.0 for j in range(k)]
        else:
            w = [math.exp(-dist / d1) for dist, _b in d]
        sw = sum(w)
        w = [t / sw for t in w]
        est = sum(w[j] * X[idx[d[j][1]]] for j in range(k))
        obs.append(X[idx[a]])
        pred.append(est)
    if len(obs) < 3:
        raise ValueError("cnvlfc: too few points survived the "
                         "embedding and Theiler window to score")
    return {"rho": _corr(obs, pred), "observed": obs,
            "predicted": pred, "n_predicted": len(obs),
            "library": len(libset), "E": int(E), "tau": int(tau)}


def ccm(x, y, E=2, tau=1, lib_sizes=None, seed=1, exclude=0):
    r"""Cross-map skill in both directions across library lengths."""
    X = [float(v) for v in x]
    Y = [float(v) for v in y]
    m = len(embed(Y, E, tau)["points"])
    if lib_sizes is None:
        base = int(E) + 3
        lib_sizes = sorted({max(base, int(m * f))
                            for f in (0.05, 0.1, 0.25, 0.5, 1.0)})
    curves = {"x_causes_y": [], "y_causes_x": []}
    for L in lib_sizes:
        # "Y cross maps X" recovers X from M_Y and supports X -> Y.
        curves["x_causes_y"].append(
            {"library": L,
             "rho": cross_map(X, Y, E, tau, L, seed, exclude)["rho"]})
        curves["y_causes_x"].append(
            {"library": L,
             "rho": cross_map(Y, X, E, tau, L, seed, exclude)["rho"]})
    out = {}
    for key in curves:
        rs = [p["rho"] for p in curves[key]]
        out[key] = {"curve": curves[key], "rho_final": rs[-1],
                    "rho_first": rs[0], "increase": rs[-1] - rs[0],
                    "converges": (rs[-1] - rs[0] > 0.05
                                  and rs[-1] > 0.3)}
    return RichResult(payload={
        "estimate": {k: out[k]["rho_final"] for k in out},
        "x_causes_y": out["x_causes_y"],
        "y_causes_x": out["y_causes_x"],
        "lib_sizes": list(lib_sizes), "E": int(E), "tau": int(tau),
        "n_embeddable": m,
        "verdict": _verdict(out),
        "method": "convergent cross mapping (Sugihara et al. 2012); "
                  "skill recovering X from the manifold of Y "
                  "supports X causing Y",
    })


def _verdict(out):
    a = out["x_causes_y"]["converges"]
    b = out["y_causes_x"]["converges"]
    if a and b:
        return ("bidirectional coupling, or synchrony -- CCM cannot "
                "separate the two")
    if a:
        return "x drives y"
    if b:
        return "y drives x"
    return "no convergent cross mapping in either direction"


def coupled_logistic(n, rx=3.8, ry=3.5, bxy=0.0, byx=0.1,
                     x0=0.4, y0=0.2, burn=300):
    r"""The system Sugihara et al. demonstrate on.

    ``bxy`` is the effect of :math:`y` on :math:`x`; ``byx`` the
    effect of :math:`x` on :math:`y`. The default is unidirectional:
    :math:`x` drives :math:`y` and not the reverse.
    """
    x, y = float(x0), float(y0)
    X, Y = [], []
    for i in range(int(n) + int(burn)):
        xn = x * (rx - rx * x - bxy * y)
        yn = y * (ry - ry * y - byx * x)
        x, y = xn, yn
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("cnvlfc: the coupled map diverged at "
                             "step %d; the parameters are outside "
                             "the bounded regime" % i)
        if i >= int(burn):
            X.append(x)
            Y.append(y)
    return {"x": X, "y": Y, "bxy": float(bxy), "byx": float(byx)}


def convergent_cross_mapping(x, y, E=2, tau=1, **kw):
    r"""Entry point: see :func:`ccm`."""
    return ccm(x, y, E, tau, **kw)
