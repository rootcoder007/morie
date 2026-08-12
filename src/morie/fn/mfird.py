"""MIRT discrimination-to-factor-loading reparameterization (Reckase 2009)."""

import math

from ._richresult import RichResult

__all__ = ["mfird", "mirt_factor_loadings"]


def mfird(a, d=None, P=None, inverse=False):
    """
    Convert MIRT discriminations to factor loadings (and back).

    Reckase (2009): the normal-ogive MIRT model and item factor
    analysis have "virtually identical statistical formulations"
    (Sec. 3.3.3); the unidimensional link is his Eq. 2.28,
    a_i = lambda_i / sqrt(1 - lambda_i^2), and the multidimensional
    normal-ogive expressions (Eqs. 6.11-6.12) carry the norming
    factor sqrt(1 + a_i P a_i'), giving

        lambda_il = a_il / sqrt(1 + a_i P a_i'),

    which reduces exactly to the inverse of Eq. 2.28 when m = 1 and
    P = 1.  With intercepts d_i, Eq. 6.12 gives the classical
    marginal proportion pi_i = Phi(d_i / sqrt(1 + a_i P a_i')), so
    the factor-analytic threshold is
    tau_i = -d_i / sqrt(1 + a_i P a_i').

    Sources
    -------
    Reckase, M. D. (2009). *Multidimensional Item Response Theory*.
    Springer, Eq. 2.28, Sec. 3.3.3, and Eqs. 6.11-6.12 (local copy
    fetched-wave3/Multidimensional_Item_Response_Theory.pdf).

    Parameters
    ----------
    a : matrix (n_items x m)
        Discrimination vectors (or, with ``inverse``, loadings).
    d : sequence of float, optional
        Intercepts d_i (normal-ogive metric); enables thresholds.
    P : matrix, optional
        Latent covariance (default identity).
    inverse : bool
        If True, treat ``a`` as loadings Lambda and return the
        discriminations a_il = lambda_il / sqrt(1 - lambda_i R
        lambda_i') (P = R, the factor correlation matrix).

    Returns
    -------
    RichResult
        Keys: loadings (or discriminations when ``inverse``),
        norming (per item sqrt factors), thresholds (when d given),
        communalities.
    """
    A = [[float(v) for v in row] for row in a]
    n = len(A)
    m = len(A[0])
    if any(len(r) != m for r in A):
        raise ValueError("a must be rectangular")
    if P is None:
        Pm = [[1.0 if i == j else 0.0 for j in range(m)]
              for i in range(m)]
    else:
        Pm = [[float(v) for v in row] for row in P]
    out = []
    norming = []
    comms = []
    for i in range(n):
        quad = sum(A[i][r] * Pm[r][c] * A[i][c]
                   for r in range(m) for c in range(m))
        if inverse:
            if quad >= 1.0:
                raise ValueError("loadings imply communality >= 1 "
                                 "(item %d)" % i)
            s = math.sqrt(1.0 - quad)
            out.append([A[i][l] / s for l in range(m)])
            norming.append(s)
            comms.append(quad)
        else:
            s = math.sqrt(1.0 + quad)
            lam = [A[i][l] / s for l in range(m)]
            out.append(lam)
            norming.append(s)
            comms.append(sum(lam[r] * Pm[r][c] * lam[c]
                             for r in range(m) for c in range(m)))
    thresholds = None
    if d is not None and not inverse:
        dv = [float(v) for v in d]
        if len(dv) != n:
            raise ValueError("need one intercept per item")
        thresholds = [-dv[i] / norming[i] for i in range(n)]
    key = "discriminations" if inverse else "loadings"
    return RichResult(payload={
        key: out,
        "norming": norming,
        "thresholds": thresholds,
        "communalities": comms,
        "inverse": bool(inverse),
        "method": "MIRT a <-> lambda (Reckase Eq. 2.28 / Eqs. 6.11-6.12)",
    })


# long descriptive alias (stub-era name)
mirt_factor_loadings = mfird


def cheatsheet():
    return "mfird: lambda = a/sqrt(1 + a P a'); tau = -d/sqrt(1 + a P a')"
