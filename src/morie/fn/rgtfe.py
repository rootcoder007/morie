# morie.fn -- function file (rootcoder007/morie)
"""Transfer function estimate."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_transfer_func_est"]


def rangayyan_transfer_func_est(x, y, fs=1.0, nperseg=None):
    r"""Transfer function and coherence (Rangayyan Ch. 3):

    .. math:: H(f) = \frac{S_{xy}(f)}{S_{xx}(f)}, \qquad
              \gamma^2(f) = \frac{|S_{xy}(f)|^2}
              {S_{xx}(f)\,S_{yy}(f)}.

    The coherence is not decoration: :math:`\gamma^2` near 1 means
    the estimate at that frequency is trustworthy, while a low value
    means noise or nonlinearity dominates and H there is meaningless.
    Both are returned together for exactly that reason. Coherence
    computed from a SINGLE segment is identically 1 at every
    frequency and says nothing, so at least two segments are required.

    Parameters
    ----------
    x, y : array-like
        Input and output signals.
    fs : float, default 1.0
        Sampling frequency.
    nperseg : int, optional
        Segment length.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``H``, ``magnitude``, ``phase``,
        ``coherence``, ``n_segments``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (transfer function estimation).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    N = x.size
    seg = max(8, N // 8) if nperseg is None else int(nperseg)
    if not 2 <= seg <= N:
        raise ValueError(f"nperseg must lie in 2..{N}, got {seg}.")
    step = seg // 2
    starts = list(range(0, N - seg + 1, step))
    if len(starts) < 2:
        raise ValueError(
            "coherence needs at least 2 segments; a single segment gives "
            "gamma^2 == 1 everywhere and is uninformative."
        )
    w = np.hanning(seg)
    Sxx = Syy = Sxy = 0.0
    for s in starts:
        X = np.fft.rfft(x[s : s + seg] * w)
        Y = np.fft.rfft(y[s : s + seg] * w)
        Sxx = Sxx + np.abs(X) ** 2
        Syy = Syy + np.abs(Y) ** 2
        Sxy = Sxy + np.conj(X) * Y
    Hf = Sxy / np.maximum(Sxx, 1e-300)
    coh = np.abs(Sxy) ** 2 / np.maximum(Sxx * Syy, 1e-300)
    return RichResult(payload={"freqs": np.fft.rfftfreq(seg, d=1.0 / fs), "H": Hf,
                               "magnitude": np.abs(Hf), "phase": np.angle(Hf),
                               "coherence": np.clip(coh, 0.0, 1.0),
                               "n_segments": len(starts),
                               "method": "H = Sxy/Sxx with coherence; low gamma^2 invalidates H"})


def cheatsheet():
    return "rgtfe: single-segment coherence is identically 1 -- refused"
