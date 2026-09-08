# morie.fn -- function file (rootcoder007/morie)
"""GANomaly anomaly score from encoder-decoder-encoder latents."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ganomscore", "adversarial_anomaly"]


def ganomscore(z, zhat, threshold=0.5):
    """GANomaly anomaly score and its min-max scaling.

    For a test sample x-hat the anomaly score is the L1 distance between
    the bottleneck code of the input, z = G_E(x-hat), and the code the
    second encoder assigns to the reconstruction, z-hat = E(G(x-hat)):

        A(x-hat) = || G_E(x-hat) - E(G(x-hat)) ||_1              (eq. 5)

    Over a test set this yields a score set S, which is rescaled to the
    unit interval before thresholding:

        s'_i = (s_i - min(S)) / (max(S) - min(S))                (eq. 6)

    Parameters
    ----------
    z : array-like, shape (m, d) or (d,)
        Bottleneck codes G_E(x) of the test samples.
    zhat : array-like, same shape as z
        Codes E(G(x)) of the reconstructions.
    threshold : float
        phi in A(x) > phi, applied to the scaled scores.

    Returns
    -------
    RichResult
        ``score``, ``scaled``, ``smin``, ``smax``, ``flagged``, ``nflag``,
        ``m``, ``d``.

    References
    ----------
    Akcay, S., Atapour-Abarghouei, A. and Breckon, T. P. (2018),
    "GANomaly: semi-supervised anomaly detection via adversarial
    training", arXiv:1805.06725.  Equations (5) and (6), read from the
    ar5iv rendering of the arXiv source (Sect. 3.3, Model Testing).
    """
    Z = C.mat(z)
    H = C.mat(zhat)
    if len(Z) != len(H) or len(Z[0]) != len(H[0]):
        raise ValueError("z and zhat must have the same shape")
    m, d = len(Z), len(Z[0])
    s = [sum(abs(Z[i][j] - H[i][j]) for j in range(d)) for i in range(m)]
    lo, hi = min(s), max(s)
    rng = hi - lo
    sc = [0.0] * m if rng == 0.0 else [(v - lo) / rng for v in s]
    flg = [1 if v > float(threshold) else 0 for v in sc]
    return RichResult(payload={
        "score": s, "scaled": sc, "smin": lo, "smax": hi,
        "flagged": flg, "nflag": sum(flg), "m": m, "d": d,
        "method": "GANomaly anomaly score (Akcay et al. 2018 eqs. 5-6)"})


adversarial_anomaly = ganomscore


def cheatsheet():
    return "adGAN: GANomaly anomaly score from encoder-decoder-encoder latents."
