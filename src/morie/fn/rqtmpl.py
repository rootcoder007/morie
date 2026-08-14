# morie.fn -- function file (rootcoder007/morie)
r"""Interval mapping of quantitative trait loci by LOD score.

**The problem with single markers.** Regressing the phenotype on one
marker confounds two things: how big the QTL effect is, and how far
the QTL sits from that marker. A large effect far away and a small
effect nearby give the same regression coefficient, so neither is
identified, and the effect is always underestimated.

**Interval mapping.** Walk a putative QTL along the interval between
two markers. Its genotype :math:`g_i` is unobserved, but the flanking
marker genotypes and the two recombination fractions give its
distribution, so the likelihood is a two-component mixture,

.. math:: L(a,b,\sigma^2) = \prod_i \big[ G_i(0)\,L_i(0)
          + G_i(1)\,L_i(1) \big], \qquad
          L_i(x) = \varphi\big(y_i - (a + bx);\ \sigma^2\big),

with :math:`G_i(x)` the probability that :math:`g_i = x` given the
flanking markers. At a marker the mixture collapses to a point mass
and the likelihood reduces to ordinary regression -- the paper says so,
and ``interval_map`` is checked against ``single_marker`` at exactly
those positions.

**The evidence.** :math:`\mathrm{LOD} = \log_{10}
L(\hat a,\hat b,\hat\sigma^2) / L(\hat a_0, 0, \hat\sigma_0^2)`. For
the Gaussian model with no missing genotypes this has the closed form
:math:`\tfrac{n}{2}\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)`, which is
what ``single_marker`` returns and what the anchor holds the
likelihood route against.

**Thresholds and sample size.** LOD is asymptotically
:math:`\tfrac12(\log_{10}e)\chi^2_1`, so a single-marker test at level
:math:`\alpha` uses :math:`T = \tfrac12(\log_{10}e)z_\alpha^2`, which
is 0.83 at 5% -- the value printed in the paper. The expected LOD per
progeny is :math:`\mathrm{ELOD} = \tfrac12\log_{10}(1 +
\sigma^2_{QTL}/\sigma^2_{res}) \approx 0.22\,
\sigma^2_{QTL}/\sigma^2_{res}`, and :math:`T/\mathrm{ELOD}` progeny
give even odds of detection. Both the exact form and the paper's
small-effect approximation are here, and ``elod`` reports the gap
between them rather than silently using one.

**Estimation.** The EM algorithm of the paper: the E step is the
posterior QTL genotype probability given the current parameters, the M
step is a weighted regression. ``interval_map`` records the
likelihood at every iteration so the monotone increase is visible --
an EM step that decreases the likelihood is a bug, and the anchor
checks for it.

**Genome-wide thresholds.** The paper is explicit that the
single-marker threshold is *not* right when many positions are
scanned. This module does not invent a genome-wide correction; use a
permutation threshold (see :mod:`morie.fn.mqtmpl`).

References
----------
Lander, E. S. & Botstein, D. (1989) "Mapping Mendelian Factors
Underlying Quantitative Traits Using RFLP Linkage Maps", *Genetics*
121(1), 185-199, doi:10.1093/genetics/121.1.185. The section "QTL
mapping: interval mapping using LOD scores": equation (4) for the
single-marker Gaussian likelihood, the definition of the LOD score,
the asymptotic :math:`\tfrac12(\log_{10}e)\chi^2_1` distribution and
the resulting threshold :math:`T = 0.83` at a 5% error rate, equations
(5a)-(5c) for the ELOD including the 0.22 approximation, equation (6)
for the number of progeny required, equation (7) for the mixture
likelihood of interval mapping with :math:`G_i(x)` from the flanking
markers, and the use of the EM algorithm to maximise it.

Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977) "Maximum
Likelihood from Incomplete Data via the EM Algorithm", *Journal of the
Royal Statistical Society. Series B* 39(1), 1-38,
doi:10.1111/j.2517-6161.1977.tb01600.x, for the algorithm itself.

Haldane, J. B. S. (1919) "The combination of linkage values, and the
calculation of distances between the loci of linked factors",
*Journal of Genetics* 8(4), 299-309, for the map function used to turn
map distance into a recombination fraction.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["haldane", "inverse_haldane", "genotype_probabilities",
           "single_marker", "interval_map", "scan_interval", "elod",
           "threshold", "progeny_required", "LOG10E"]

LOG10E = math.log10(math.e)


def haldane(distance):
    r"""Map distance in Morgans to a recombination fraction."""
    d = float(distance)
    if d < 0.0:
        raise ValueError("rqtmpl: map distance cannot be negative")
    return 0.5 * (1.0 - math.exp(-2.0 * d))


def inverse_haldane(r):
    r"""Recombination fraction back to Morgans."""
    r = float(r)
    if not 0.0 <= r < 0.5:
        raise ValueError("rqtmpl: a recombination fraction must lie "
                         "in [0, 0.5), got %r" % r)
    return -0.5 * math.log(1.0 - 2.0 * r)


def genotype_probabilities(left, right, r_left, r_right):
    r""":math:`G_i(x)`: the QTL genotype given the flanking markers.

    Backcross coding, 0 or 1 at every locus, no interference, so the
    two intervals contribute independently and the result is the
    normalised product of the two recombination probabilities.
    """
    for r in (r_left, r_right):
        if not 0.0 <= float(r) <= 0.5:
            raise ValueError("rqtmpl: recombination fractions lie in "
                             "[0, 0.5], got %r" % r)
    out = []
    for q in (0, 1):
        p = (float(r_left) if q != int(left) else 1.0 - float(r_left))
        p *= (float(r_right) if int(right) != q
              else 1.0 - float(r_right))
        out.append(p)
    tot = out[0] + out[1]
    if tot <= 0.0:
        raise ValueError("rqtmpl: the flanking marker configuration "
                         "has probability zero")
    return [out[0] / tot, out[1] / tot]


def _normal_ll(resid, sigma2):
    n = len(resid)
    return (-0.5 * n * math.log(2.0 * math.pi * sigma2)
            - sum(v * v for v in resid) / (2.0 * sigma2))


def single_marker(y, g):
    r"""Regression of the phenotype on one marker genotype.

    Returns the LOD both from the likelihood ratio and from the closed
    form :math:`\tfrac{n}{2}\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)`.
    """
    n = len(y)
    if n != len(g):
        raise ValueError("rqtmpl: y and g must have the same length")
    if n < 3:
        raise ValueError("rqtmpl: need at least three individuals")
    gs = [float(v) for v in g]
    if max(gs) == min(gs):
        raise ValueError("rqtmpl: the marker is monomorphic, so no "
                         "effect is identified")
    my = sum(y) / n
    mg = sum(gs) / n
    b = (sum((gs[i] - mg) * (y[i] - my) for i in range(n))
         / sum((v - mg) ** 2 for v in gs))
    a = my - b * mg
    r1 = [y[i] - (a + b * gs[i]) for i in range(n)]
    r0 = [y[i] - my for i in range(n)]
    rss1 = sum(v * v for v in r1)
    rss0 = sum(v * v for v in r0)
    s1 = rss1 / n
    s0 = rss0 / n
    lod_lr = (_normal_ll(r1, s1) - _normal_ll(r0, s0)) * LOG10E
    lod_cf = 0.5 * n * math.log10(rss0 / rss1) if rss1 > 0.0 \
        else float("inf")
    return RichResult(payload={
        "estimate": lod_cf, "lod": lod_cf, "lod_likelihood": lod_lr,
        "a": a, "b": b, "sigma2": s1, "rss": rss1, "rss_null": rss0,
        "n": n,
        "method": "single-marker regression LOD; Lander & Botstein "
                  "(1989) eq (4)",
    })


def interval_map(y, left, right, r_left, r_right, max_iter=200,
                 tol=1e-10):
    r"""EM for the mixture likelihood (7) at one QTL position."""
    n = len(y)
    if not (n == len(left) == len(right)):
        raise ValueError("rqtmpl: y and the two marker vectors must "
                         "have the same length")
    G = [genotype_probabilities(left[i], right[i], r_left, r_right)
         for i in range(n)]
    my = sum(y) / n
    a = my
    b = 0.1 * (max(y) - min(y) + 1e-12)
    s2 = sum((v - my) ** 2 for v in y) / n
    history = []
    for _ in range(int(max_iter)):
        post = []
        ll = 0.0
        for i in range(n):
            d0 = math.exp(-((y[i] - a) ** 2) / (2.0 * s2))
            d1 = math.exp(-((y[i] - (a + b)) ** 2) / (2.0 * s2))
            m0 = G[i][0] * d0
            m1 = G[i][1] * d1
            tot = m0 + m1
            if tot <= 0.0:
                raise ValueError("rqtmpl: the mixture vanished at "
                                 "individual %d" % i)
            post.append(m1 / tot)
            ll += math.log(tot / math.sqrt(2.0 * math.pi * s2))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
        sw = sum(post)
        if sw <= 0.0 or sw >= n:
            b_new = 0.0
            a_new = my
        else:
            a_new = (sum(y[i] * (1.0 - post[i]) for i in range(n))
                     / (n - sw))
            a_plus_b = sum(y[i] * post[i] for i in range(n)) / sw
            b_new = a_plus_b - a_new
        s2 = sum((1.0 - post[i]) * (y[i] - a_new) ** 2
                 + post[i] * (y[i] - (a_new + b_new)) ** 2
                 for i in range(n)) / n
        a, b = a_new, b_new
    s0 = sum((v - my) ** 2 for v in y) / n
    ll0 = -0.5 * n * (math.log(2.0 * math.pi * s0) + 1.0)
    lod = (history[-1] - ll0) * LOG10E
    return RichResult(payload={
        "estimate": lod, "lod": lod, "a": a, "b": b, "sigma2": s2,
        "loglik": history[-1], "loglik_null": ll0,
        "iterations": len(history), "loglik_history": history,
        "posterior": post, "n": n,
        "method": "interval mapping by EM on the mixture likelihood; "
                  "Lander & Botstein (1989) eq (7)",
    })


def scan_interval(y, left, right, length, step=0.01, **kw):
    r"""LOD along an interval of the given length, in Morgans."""
    length = float(length)
    if length <= 0.0:
        raise ValueError("rqtmpl: the interval length must be "
                         "positive")
    positions, lods, fits = [], [], []
    d = 0.0
    while d <= length + 1e-12:
        r1 = haldane(min(d, length))
        r2 = haldane(max(length - d, 0.0))
        f = interval_map(y, left, right, r1, r2, **kw)
        positions.append(d)
        lods.append(f["lod"])
        fits.append(f)
        d += float(step)
    k = max(range(len(lods)), key=lambda i: lods[i])
    return RichResult(payload={
        "estimate": lods[k], "peak_lod": lods[k],
        "peak_position": positions[k], "position": positions,
        "lod": lods, "fit": fits[k],
        "method": "interval scan; Lander & Botstein (1989)",
    })


def elod(var_qtl, var_residual):
    r"""Equations (5a)-(5c): expected LOD per progeny."""
    vq, vr = float(var_qtl), float(var_residual)
    if vr <= 0.0:
        raise ValueError("rqtmpl: the residual variance must be "
                         "positive")
    if vq < 0.0:
        raise ValueError("rqtmpl: a variance cannot be negative")
    exact = 0.5 * math.log10(1.0 + vq / vr)
    approx = 0.22 * (vq / vr)
    return {"elod": exact, "approximation": approx,
            "gap": approx - exact, "ratio": vq / vr,
            "note": "0.22 = (1/2) log10(e); the approximation is a "
                    "Taylor expansion and drifts upward as the "
                    "effect grows"}


def threshold(alpha=0.05):
    r""":math:`T = \tfrac12(\log_{10}e)z_\alpha^2` for one marker."""
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("rqtmpl: alpha must lie in (0, 1)")
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if math.erfc(mid / math.sqrt(2.0)) > a:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    return {"threshold": 0.5 * LOG10E * z * z, "z": z, "alpha": a,
            "note": "single-marker only; a genome scan needs a "
                    "permutation threshold"}


def progeny_required(var_qtl, var_residual, alpha=0.05):
    r"""Equation (6): :math:`T/\mathrm{ELOD}` progeny."""
    t = threshold(alpha)["threshold"]
    e = elod(var_qtl, var_residual)["elod"]
    if e <= 0.0:
        raise ValueError("rqtmpl: a QTL with no variance is never "
                         "detected")
    return {"n": t / e, "threshold": t, "elod": e}


def cheatsheet():
    return ("rqtmpl: interval mapping walks a QTL along an interval "
            "and maximises the MIXTURE likelihood (7) by EM, because "
            "the QTL genotype is unknown -- G_i(x) comes from the "
            "flanking markers. LOD = log10 of the likelihood ratio, "
            "and at a marker it collapses to the single-marker "
            "regression LOD (n/2) log10(RSS0/RSS1). T = 0.83 at 5% "
            "for ONE marker; a genome scan needs permutations. "
            "ELOD = (1/2) log10(1 + var_qtl/var_res), with the "
            "paper's 0.22 approximation kept alongside it.")


# compact alias per ledger/NAMING.md
interval_mapping = scan_interval
