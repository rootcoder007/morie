# morie.fn -- function file (rootcoder007/morie)
"""Sample (empirical) semivariogram."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["svariog", "bivand2013_chapter_8_equation_4"]


def svariog(coords, z, breaks=None, nbins=10, cutoff=None):
    """Matheron's method-of-moments estimate of the semivariogram.

    Under intrinsic stationarity the semivariance depends only on the
    separation vector, gamma(h) = (1/2) E(Z(s) - Z(s+h))^2, and with
    isotropy only on its length.  Pairs of observations are therefore
    grouped into distance classes and the semivariance estimated within
    each class by

        gammahat(hbar_j) = (1 / (2 N_h)) sum_{i=1}^{N_h}
                           ( Z(s_i) - Z(s_i + h) )^2,   for all h in hbar_j

    with N_h the number of pairs falling in class j.  The reported
    distance for a class is the mean separation of the pairs in it, which
    is what a variogram plot puts on the horizontal axis.

    Parameters
    ----------
    coords : array-like, shape (n, k)
        Locations, one row per observation.
    z : array-like
        Observed values, length n.  For a varying-mean model these should
        be regression residuals, not the raw data.
    breaks : array-like or None
        Increasing distance-class boundaries.  ``None`` builds ``nbins``
        equal-width classes from 0 to ``cutoff``.
    nbins : int
        Number of classes when ``breaks`` is not given.
    cutoff : float or None
        Largest separation used; ``None`` takes one third of the largest
        interpoint distance, the usual default.

    Returns
    -------
    RichResult
        ``gamma``, ``np``, ``dist``, ``breaks``, ``cutoff``, ``n``,
        ``npair``.

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013),
    Applied Spatial Data Analysis with R, 2nd edn, Springer (Use R!).  Equation (8.4), p. 218: gammahat(hbar_j) = (1/(2 N_h)) sum_i
    (Z(s_i) - Z(s_i + h))^2 for all h in hbar_j, called there the sample
    variogram; Equation (8.3) on the same page is the semivariance it
    estimates, and the surrounding text is explicit that for a varying
    mean the sample variogram must be computed from estimated residuals.
    Read from the corpus PDF (bivand2013.pdf, p. 218).
    """
    P = C.mat(coords)
    z = C.vec(z)
    n = len(P)
    if len(z) != n:
        raise ValueError("z must have one value per location")
    if n < 2:
        raise ValueError("need at least two observations")
    k = len(P[0])
    d = []
    g = []
    for i in range(n):
        for j in range(i + 1, n):
            dd = math.sqrt(sum((P[i][t] - P[j][t]) ** 2 for t in range(k)))
            d.append(dd)
            g.append((z[i] - z[j]) ** 2)
    if cutoff is None:
        cut = max(d) / 3.0
    else:
        cut = float(cutoff)
    if breaks is None:
        nb = int(nbins)
        if nb < 1:
            raise ValueError("nbins must be positive")
        br = [cut * t / nb for t in range(nb + 1)]
    else:
        br = C.vec(breaks)
        if any(br[t + 1] <= br[t] for t in range(len(br) - 1)):
            raise ValueError("breaks must be strictly increasing")
        nb = len(br) - 1
    ssq = [0.0] * nb
    sdi = [0.0] * nb
    cnt = [0] * nb
    for t in range(len(d)):
        if d[t] <= br[0] or d[t] > br[nb]:
            continue
        b = 0
        while b < nb - 1 and d[t] > br[b + 1]:
            b += 1
        ssq[b] += g[t]
        sdi[b] += d[t]
        cnt[b] += 1
    gam = [ssq[b] / (2.0 * cnt[b]) if cnt[b] else float("nan")
           for b in range(nb)]
    dis = [sdi[b] / cnt[b] if cnt[b] else float("nan") for b in range(nb)]
    return RichResult(payload={
        "gamma": gam, "np": cnt, "dist": dis, "breaks": br, "cutoff": cut,
        "n": n, "npair": len(d),
        "method": "Sample semivariogram (Bivand et al. 2013 eq. 8.4)"})


bivand2013_chapter_8_equation_4 = svariog


def cheatsheet():
    return "bivand20138e4: Sample (empirical) semivariogram."
