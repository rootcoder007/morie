# morie.fn -- function file (rootcoder007/morie)
"""Random orthogonal rotation by QR of a Gaussian matrix."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_rotation_matrix"]


def turboquant_rotation_matrix(d, seed=1):
    """Draw a rotation that spreads outlier energy over all coordinates.

    Quantizers do badly when a few coordinates carry most of the norm,
    and key embeddings are exactly like that.  Rotating first makes the
    coordinates exchangeable, so per-coordinate error budgets stop being
    wrong.  The rotation must be genuinely orthogonal, not merely
    Gaussian, or it would change the norms it is supposed to preserve.

    Determinism: the Gaussian entries come from the shared Lehmer
    minstd stream, and the factorisation is modified Gram-Schmidt, whose
    ``R`` diagonal is non-negative by construction.  Both arms therefore
    return the same ``Q`` bit for bit -- a LAPACK-vs-LINPACK QR would
    not.

    Formula: ``A ~ N(0, 1)^{d x d}``, ``Q, R = QR(A)``, return ``Q``.

    Parameters
    ----------
    d : int
        Dimension.
    seed : int, default 1
        Seed for the shared generator.

    Returns
    -------
    RichResult
        ``Q``, ``estimate`` (``Q[0][0]``), ``d``, ``orth_err`` (the
        largest absolute deviation of ``Q' Q`` from the identity).

    References
    ----------
    Zandieh, A., Daliri, M. & Han, I. (2024).  QJL: 1-bit quantized
    JL transform for KV cache quantization with zero overhead.
    arXiv:2406.03482.  Fetched and read; the definitions and bounds used
    here are that paper own (definition 3.1, fact 3.4, lemma 3.5,
    theorem 3.6).  The KV-cache system built on it is Zandieh, A. et al.
    (2025), TurboQuant: online vector quantization with near-optimal
    distortion rate, arXiv:2504.19874.
    """
    d = int(d)
    g = C.Lcg(seed)
    A = [[g.norm() for _ in range(d)] for _ in range(d)]
    Q, _ = S.qr_mgs(A)
    err = 0.0
    for i in range(d):
        for j in range(d):
            v = sum(Q[r][i] * Q[r][j] for r in range(d)) - (1.0 if i == j else 0.0)
            if abs(v) > err:
                err = abs(v)
    return RichResult(payload={
        "Q": Q, "estimate": Q[0][0], "d": d, "orth_err": err,
        "method": "Random orthogonal rotation, QR of a Gaussian matrix"})


def cheatsheet():
    return "tqrot: Random orthogonal rotation by QR of a Gaussian matrix."
