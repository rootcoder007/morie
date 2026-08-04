# morie.fn -- function file (rootcoder007/morie)
"""NTK-scaled dynamic RoPE."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["rope_ntk_dynamic"]


def rope_ntk_dynamic(y, q, m, theta=10000.0, L_new=None, L_train=None):
    """Rotary position embedding rescaled to reach past the trained length.

    Interpolating every frequency by the same factor stretches the
    high-frequency dimensions too, and those are the ones carrying local
    word order, so plain position interpolation blurs exactly the
    information attention needs at short range.  The NTK-aware variant
    scales the base instead: low frequencies stretch a lot, high
    frequencies barely at all, so long-range reach is bought without
    destroying local resolution.

    Formula: ``theta_i' = theta (L_new / L_train)^(d / (d - 2))`` applied
    inside ``theta_i = theta'^(-2i/d)``, then the usual rotation
    ``[q_2i, q_2i+1] -> [q cos - q' sin, q sin + q' cos]`` at angle
    ``m theta_i``.

    Parameters
    ----------
    y : array-like
        Unused; kept so the signature matches the surrounding family.
    q : array-like, shape (d,)
        Query or key vector, ``d`` even.
    m : float
        Position index.
    theta : float, default 10000.0
        Base.
    L_new, L_train : float, optional
        Target and trained context lengths; no rescaling when either is
        absent.

    Returns
    -------
    RichResult
        ``estimate`` (rotated vector), ``theta_base``, ``freqs``,
        ``scale``, ``d``.

    References
    ----------
    The rotation is Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B. &
    Liu, Y. (2024), RoFormer: enhanced transformer with rotary position
    embedding, Neurocomputing 568:127063.  The NTK-aware base rescaling
    is bloc97 (2023), NTK-aware scaled RoPE, published as a LocalLLaMA
    community note rather than a paper; it is cited here as such
    because no peer-reviewed source for it exists.
    """
    qv = C.vec(q)
    d = len(qv)
    scale = 1.0
    base = float(theta)
    if L_new is not None and L_train is not None and L_train > 0:
        a = float(L_new) / float(L_train)
        scale = a ** (d / (d - 2.0)) if d > 2 else a
        base = base * scale
    freqs = [base ** (-2.0 * i / d) for i in range(d // 2)]
    out = list(qv)
    for i in range(d // 2):
        ang = float(m) * freqs[i]
        c_, s_ = math.cos(ang), math.sin(ang)
        a0, a1 = qv[2 * i], qv[2 * i + 1]
        out[2 * i] = a0 * c_ - a1 * s_
        out[2 * i + 1] = a0 * s_ + a1 * c_
    return RichResult(payload={
        "estimate": out, "theta_base": base, "freqs": freqs, "scale": scale,
        "d": d, "method": "NTK-aware dynamically scaled RoPE"})


ropentkdynamic = rope_ntk_dynamic


def cheatsheet():
    return "ropedy: NTK-scaled dynamic RoPE."
