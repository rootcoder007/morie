# morie.fn -- function file (rootcoder007/morie)
r"""Composite interval mapping: interval mapping with marker cofactors.

**What plain interval mapping gets wrong.** Its likelihood knows about
one QTL. Put two linked QTLs on a chromosome and the profile between
them rises even where nothing is -- a ghost peak -- because a position
in the middle explains part of both. Unlinked QTLs are not modelled
either, so their variance sits in the residual and costs power.

**Zeng's fix.** Test the interval :math:`(i, i+1)` while regressing
out the other markers as cofactors,

.. math:: y_j = b_0 + b^* x_j^* + \sum_{k \ne i, i+1} b_k x_{jk}
          + \epsilon_j ,

with :math:`x_j^*` the unobserved QTL indicator, so the likelihood is
the mixture

.. math:: L = \prod_j \big[ p_j(1) f_j(1) + p_j(0) f_j(0) \big] .

Four properties from the paper drive the design and each has a visible
consequence here:

*Property 1.* The expected partial regression coefficient on a marker
depends only on QTLs inside the interval its two neighbours bracket.
So conditioning on the other markers turns the scan into a *local*
test -- ``scan`` shows the ghost peak collapsing.

*Property 2.* Conditioning on **unlinked** markers cuts the residual
variance and so raises power. ``scan(..., cofactors=...)`` with an
unlinked marker demonstrably raises the LOD at the true position.

*Property 3.* Conditioning on **linked** markers buys precision at the
cost of sampling variance -- a trade-off, not a free lunch, which is
why ``window`` exists: cofactors within the window of the tested
interval are dropped so they do not absorb the very effect being
tested.

*Property 4.* Test statistics in non-adjacent intervals are
essentially uncorrelated, which is what makes a scan interpretable
position by position.

**No cofactors is exactly interval mapping.** ``cim`` with an empty
cofactor set reproduces :func:`morie.fn.rqtmpl.interval_map` to
machine precision, and the anchor checks it.

References
----------
Zeng, Z.-B. (1994) "Precision Mapping of Quantitative Trait Loci",
*Genetics* 136(4), 1457-1468, doi:10.1093/genetics/136.4.1457.
Equation (5) for the composite model with the putative QTL indicator
and the marker cofactors, equation (6) for the mixture likelihood with
the prior probabilities :math:`p_j(1)` and :math:`p_j(0)` from Table 1,
and Properties 1-4 with their consequences for a conditional interval
test, for power under unlinked cofactors, for the precision-efficiency
trade-off under linked cofactors, and for the correlation between
tests in different intervals.

Zeng, Z.-B. (1993) "Theoretical basis for separation of multiple
linked gene effects in mapping quantitative trait loci", *Proceedings
of the National Academy of Sciences USA* 90(23), 10972-10976,
doi:10.1073/pnas.90.23.10972, for the partial-regression argument the
properties rest on.

Lander, E. S. & Botstein, D. (1989) "Mapping Mendelian Factors
Underlying Quantitative Traits Using RFLP Linkage Maps", *Genetics*
121(1), 185-199, doi:10.1093/genetics/121.1.185, for the interval
mapping this extends and reduces to.
"""

import math

from . import _array_core as np
from . import rqtmpl as _im
from ._richresult import RichResult

__all__ = ["cim", "scan", "select_cofactors", "ghost_peak_demo"]


def _wls(X, y, w):
    """Weighted least squares with an intercept already in X."""
    p = len(X[0])
    A = [[sum(w[i] * X[i][r] * X[i][c] for i in range(len(y)))
          for c in range(p)] for r in range(p)]
    b = [sum(w[i] * X[i][r] * y[i] for i in range(len(y)))
         for r in range(p)]
    return [float(v) for v in np.linalg.solve(np.array(A),
                                              np.array(b))]


def cim(y, left, right, r_left, r_right, cofactors=(), max_iter=200,
        tol=1e-10):
    r"""EM for the composite likelihood (6) at one QTL position.

    ``cofactors`` is a list of marker genotype vectors to condition
    on. An empty list is exactly interval mapping.
    """
    n = len(y)
    if not (n == len(left) == len(right)):
        raise ValueError("cqtmpl: y and the flanking markers must "
                         "have the same length")
    cof = [[float(v) for v in c] for c in cofactors]
    for c in cof:
        if len(c) != n:
            raise ValueError("cqtmpl: every cofactor must have %d "
                             "entries" % n)
    G = [_im.genotype_probabilities(left[i], right[i], r_left,
                                    r_right) for i in range(n)]
    my = sum(y) / n
    beta = [my, 0.1 * (max(y) - min(y) + 1e-12)] + [0.0] * len(cof)
    s2 = sum((v - my) ** 2 for v in y) / n
    history = []
    post = [0.5] * n

    def mean_at(i, q):
        m = beta[0] + beta[1] * q
        for k in range(len(cof)):
            m += beta[2 + k] * cof[k][i]
        return m

    for _ in range(int(max_iter)):
        ll = 0.0
        for i in range(n):
            d0 = math.exp(-((y[i] - mean_at(i, 0.0)) ** 2)
                          / (2.0 * s2))
            d1 = math.exp(-((y[i] - mean_at(i, 1.0)) ** 2)
                          / (2.0 * s2))
            m0, m1 = G[i][0] * d0, G[i][1] * d1
            tot = m0 + m1
            if tot <= 0.0:
                raise ValueError("cqtmpl: the mixture vanished at "
                                 "individual %d" % i)
            post[i] = m1 / tot
            ll += math.log(tot / math.sqrt(2.0 * math.pi * s2))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
        # M step: stack each individual twice, weighted by the
        # posterior, and run one weighted regression.
        X, Y, W = [], [], []
        for i in range(n):
            for q, w in ((0.0, 1.0 - post[i]), (1.0, post[i])):
                X.append([1.0, q] + [cof[k][i]
                                     for k in range(len(cof))])
                Y.append(y[i])
                W.append(w)
        beta = _wls(X, Y, W)
        s2 = sum(W[j] * (Y[j] - sum(beta[t] * X[j][t]
                                    for t in range(len(beta)))) ** 2
                 for j in range(len(Y))) / n
    # null model: the same cofactors, no QTL term
    X0 = [[1.0] + [cof[k][i] for k in range(len(cof))]
          for i in range(n)]
    b0 = _wls(X0, list(y), [1.0] * n)
    r0 = [y[i] - sum(b0[t] * X0[i][t] for t in range(len(b0)))
          for i in range(n)]
    s0 = sum(v * v for v in r0) / n
    ll0 = -0.5 * n * (math.log(2.0 * math.pi * s0) + 1.0)
    lod = (history[-1] - ll0) * _im.LOG10E
    return RichResult(payload={
        "estimate": lod, "lod": lod, "b0": beta[0], "b": beta[1],
        "cofactor_coefficients": beta[2:], "sigma2": s2,
        "sigma2_null": s0, "loglik": history[-1], "loglik_null": ll0,
        "iterations": len(history), "loglik_history": history,
        "posterior": list(post), "n_cofactors": len(cof), "n": n,
        "method": "composite interval mapping by EM; Zeng (1994) "
                  "eqs (5)-(6)",
    })


def select_cofactors(y, markers, k=5):
    r"""Forward selection of marker cofactors by residual sum of
    squares."""
    n = len(y)
    pool = list(range(len(markers)))
    chosen = []
    for _ in range(min(int(k), len(pool))):
        best = None
        for j in pool:
            cols = chosen + [j]
            X = [[1.0] + [float(markers[c][i]) for c in cols]
                 for i in range(n)]
            try:
                b = _wls(X, list(y), [1.0] * n)
            except Exception:
                continue
            rss = sum((y[i] - sum(b[t] * X[i][t]
                                  for t in range(len(b)))) ** 2
                      for i in range(n))
            if best is None or rss < best[0]:
                best = (rss, j)
        if best is None:
            break
        chosen.append(best[1])
        pool.remove(best[1])
    return {"cofactors": chosen,
            "note": "forward selection; Zeng (1994) leaves the "
                    "cofactor count to the analyst, trading Property "
                    "2 against Property 3"}


def _window_filter(marker_positions, cofactor_index, interval_centre,
                   window):
    return [j for j in cofactor_index
            if abs(marker_positions[j] - interval_centre) > window]


def scan(y, markers, positions, cofactors=(), window=0.10, step=0.01,
         **kw):
    r"""CIM profile along a chromosome of ordered markers.

    ``window`` (in Morgans) drops any cofactor too close to the
    interval being tested -- Property 3's trade-off made explicit.
    """
    m = len(markers)
    if m < 2:
        raise ValueError("cqtmpl: need at least two markers")
    if len(positions) != m:
        raise ValueError("cqtmpl: one position per marker")
    cof_idx = list(cofactors)
    out_pos, out_lod, fits = [], [], []
    for j in range(m - 1):
        span = float(positions[j + 1]) - float(positions[j])
        if span <= 0.0:
            raise ValueError("cqtmpl: marker positions must increase")
        d = 0.0
        while d <= span + 1e-12:
            centre = float(positions[j]) + d
            keep = _window_filter(positions, cof_idx, centre, window)
            keep = [c for c in keep if c not in (j, j + 1)]
            f = cim(y, markers[j], markers[j + 1],
                    _im.haldane(min(d, span)),
                    _im.haldane(max(span - d, 0.0)),
                    [markers[c] for c in keep], **kw)
            out_pos.append(centre)
            out_lod.append(f["lod"])
            fits.append(f)
            d += float(step)
    k = max(range(len(out_lod)), key=lambda i: out_lod[i])
    return RichResult(payload={
        "estimate": out_lod[k], "peak_lod": out_lod[k],
        "peak_position": out_pos[k], "position": out_pos,
        "lod": out_lod, "fit": fits[k], "window": float(window),
        "n_cofactors": len(cof_idx),
        "method": "composite interval mapping scan; Zeng (1994)",
    })


def ghost_peak_demo(y, markers, positions, cofactors, window=0.10,
                    step=0.02):
    r"""The same chromosome scanned with and without cofactors.

    Returns both profiles so the ghost peak of plain interval mapping
    can be compared with the composite scan rather than described.
    """
    plain = scan(y, markers, positions, (), window, step)
    comp = scan(y, markers, positions, cofactors, window, step)
    return {"interval_mapping": plain, "composite": comp,
            "peak_shift": comp["peak_position"]
            - plain["peak_position"]}


def cheatsheet():
    return ("cqtmpl: CIM = interval mapping with the other markers as "
            "cofactors, so the test at an interval becomes LOCAL "
            "(Property 1). Unlinked cofactors cut residual variance "
            "and raise power (Property 2); linked ones buy precision "
            "at the cost of variance (Property 3), which is what the "
            "window parameter manages. With no cofactors it IS "
            "interval mapping, to machine precision.")


# compact alias per ledger/NAMING.md
composite_interval_mapping = scan
