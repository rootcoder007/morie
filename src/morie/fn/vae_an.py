# morie.fn -- function file (rootcoder007/morie)
"""Anomaly detection by VAE reconstruction probability.

SOURCE.  An, J. and Cho, S. (2015), "Variational Autoencoder based
Anomaly Detection using Reconstruction Probability", *Special Lecture on
IE* 2:1-18, SNU Data Mining Center.

Their Algorithm 4 is the whole method.  For each test point x_i:

  1. (mu_z, sigma_z) = encoder(x_i)
  2. draw z^(1) ... z^(L) from N(mu_z, diag(sigma_z^2))
  3. (mu_x^(l), sigma_x^(l)) = decoder(z^(l))
  4. reconstruction probability = (1/L) sum_l p( x_i | mu_x^(l), sigma_x^(l) )
  5. flag x_i as an anomaly when that probability is below a threshold.

The paper's point (their Section 4) is that this is a *probability*, not
a reconstruction error: the decoder's variance makes the score
comparable across dimensions with different scales, which a squared
error is not.

MODEL.  A reference implementation cannot ship trained weights, and an
untrained autoencoder detects nothing, so the default encoder/decoder
are the LINEAR-GAUSSIAN instance, whose optimum is available in closed
form: the maximum-likelihood decoder subspace of a linear-Gaussian
latent model is the principal subspace of the data, and the residual
variance is the mean of the discarded eigenvalues -- Tipping, M.E. and
Bishop, C.M. (1999), "Probabilistic Principal Component Analysis",
*JRSS-B* 61(3):611-622, doi:10.1111/1467-9868.00196, Section 3.2.  So

    W       = top-k eigenvectors of the sample covariance
    mu_z(x) = W' (x - xbar),        sigma_z = ``encoder_sd``
    mu_x(z) = W z + xbar,           sigma_x = ``decoder_scale``

with decoder_scale defaulting to the Tipping-Bishop residual variance.
That default makes the module a working detector rather than a shell;
passing ``vae`` explicitly replaces it.  Using the closed-form optimum
in place of stochastic training is this implementation's choice, stated
rather than attributed.

z^(l) is drawn from the shared deterministic normal stream, so both
language arms hold the same draws.

THRESHOLD.  ``alpha`` is a tail fraction: the cut is the type-7
``alpha``-quantile of the log reconstruction probabilities unless
``threshold`` is given.  The decision is therefore relative to the
sample, which is what makes the all-inlier and all-outlier degenerate
cases behave sensibly.

ANCHOR.  With ``latent_dim`` equal to the data dimension and
``encoder_sd = 0`` the map W W' is the identity, reconstruction is
exact, and the log reconstruction probability is exactly
-d/2 * log(2 pi s^2) for every point.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vae_anomaly"]


def vae_anomaly(X, vae=None, latent_dim=1, n_samples=32, alpha=0.1,
                encoder_sd=0.0, decoder_scale=None, threshold=None, skip=0):
    """Reconstruction-probability anomaly score and decision.

    Parameters
    ----------
    X : array-like
        n-by-d data matrix.
    vae : mapping or None
        ``{"W": d-by-k, "center": length d}``; ``None`` uses the
        closed-form principal subspace.
    latent_dim : int
        k, 1 <= k <= d, used only when ``vae`` is ``None``.
    n_samples : int
        L, the number of latent draws per point.
    alpha : float
        Tail fraction in [0, 1] defining the threshold quantile.
    encoder_sd : float
        sigma_z, >= 0.
    decoder_scale : float or None
        sigma_x, > 0.  ``None`` uses the Tipping-Bishop residual
        variance (1 when k = d).
    threshold : float or None
        Explicit cut on the log reconstruction probability.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    RichResult
        ``reconstruction_probability``, ``log_rp``, ``anomaly`` (0/1),
        ``threshold``, ``n_anomalies``, ``W``, ``center``,
        ``decoder_scale``, ``eigenvalues``, ``n``, ``d``,
        ``latent_dim``, ``n_samples``.

    Raises
    ------
    ValueError
        Empty or ragged ``X``, ``latent_dim`` outside 1..d, a
        non-positive ``n_samples`` or ``decoder_scale``, a negative
        ``encoder_sd``, or ``alpha`` outside [0, 1].

    References
    ----------
    An, J. and Cho, S. (2015).  Special Lecture on IE 2:1-18.
    Tipping, M.E. and Bishop, C.M. (1999).  JRSS-B 61(3):611-622.
    doi:10.1111/1467-9868.00196.
    """
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("vae_anomaly: X is empty")
    d = len(A[0])
    for r in A:
        if len(r) != d:
            raise ValueError("vae_anomaly: rows of X have unequal length")
    L = int(n_samples)
    if L < 1:
        raise ValueError("vae_anomaly: n_samples must be positive")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("vae_anomaly: alpha must lie in [0, 1]")
    esd = float(encoder_sd)
    if esd < 0.0:
        raise ValueError("vae_anomaly: encoder_sd must be non-negative")
    skip = int(skip)
    if skip < 0:
        raise ValueError("vae_anomaly: skip must be non-negative")
    cen = [0.0] * d
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += A[i][j]
        cen[j] = s / n
    C = [[0.0] * d for _ in range(d)]
    for a in range(d):
        for b in range(d):
            s = 0.0
            for i in range(n):
                s += (A[i][a] - cen[a]) * (A[i][b] - cen[b])
            C[a][b] = s / n
    ev, evec = core.jacobi(C)
    if vae is None:
        k = int(latent_dim)
        if k < 1 or k > d:
            raise ValueError("vae_anomaly: latent_dim must lie in 1 .. d")
        # eigenvalues ascend: the top k are the LAST k columns, taken
        # largest-first so column 1 is the leading direction.
        W = [[evec[i][d - 1 - t] for t in range(k)] for i in range(d)]
    else:
        W = core.mat(vae["W"])
        if len(W) != d:
            raise ValueError("vae_anomaly: vae W must have d rows")
        k = len(W[0])
        if "center" in vae:
            cen = core.vec(vae["center"])
            if len(cen) != d:
                raise ValueError("vae_anomaly: vae center must have length d")
    if decoder_scale is None:
        rest = 0.0
        cnt = 0
        for t in range(d - k):
            rest += ev[t]
            cnt += 1
        s = math.sqrt(rest / cnt) if cnt > 0 and rest > 0.0 else 1.0
    else:
        s = float(decoder_scale)
    if not (s > 0.0):
        raise ValueError("vae_anomaly: decoder_scale must be positive")
    eps = vc.draw(L, k, skip, 1.0)
    c = math.log(2.0 * math.pi * s * s)
    lrp = [0.0] * n
    rp = [0.0] * n
    for i in range(n):
        mz = [0.0] * k
        for t in range(k):
            v = 0.0
            for j in range(d):
                v += W[j][t] * (A[i][j] - cen[j])
            mz[t] = v
        ll = [0.0] * L
        best = None
        for l in range(L):
            z = [mz[t] + esd * eps[l][t] for t in range(k)]
            q = 0.0
            for j in range(d):
                r = cen[j]
                for t in range(k):
                    r += W[j][t] * z[t]
                q += c + (A[i][j] - r) * (A[i][j] - r) / (s * s)
            ll[l] = -0.5 * q
            if best is None or ll[l] > best:
                best = ll[l]
        acc = 0.0
        for l in range(L):
            acc += math.exp(ll[l] - best)
        lrp[i] = best + math.log(acc / L)
        rp[i] = math.exp(lrp[i])
    cut = core.quantile7(lrp, alpha) if threshold is None else float(threshold)
    flag = [1.0 if lrp[i] < cut else 0.0 for i in range(n)]
    nan_ = 0
    for v in flag:
        nan_ += int(v)
    return RichResult(
        title="VAE reconstruction-probability anomaly detection",
        summary_lines=[("obs", n), ("flagged", nan_), ("threshold", cut)],
        payload={
            "estimate": cut,
            "reconstruction_probability": rp,
            "log_rp": lrp,
            "anomaly": flag,
            "threshold": cut,
            "n_anomalies": nan_,
            "W": W,
            "center": cen,
            "decoder_scale": s,
            "eigenvalues": ev,
            "n": n,
            "d": d,
            "latent_dim": k,
            "n_samples": L,
            "method": "Reconstruction probability, An and Cho (2015) Algorithm 4, on the closed-form linear-Gaussian optimum (Tipping and Bishop 1999 Sec. 3.2)",
        },
    )


def cheatsheet():
    return "vae_an: VAE reconstruction-probability anomaly detection (An & Cho 2015)"
