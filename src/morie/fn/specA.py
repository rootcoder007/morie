# morie.fn -- function file (rootcoder007/morie)
"""Spectral residual saliency (anomaly detection)."""

from math import fsum, log, sqrt

from ._richresult import RichResult
from ._spx import dft, idftre, phase, vec

__all__ = [
    "spectral_anomaly",
    "specanom",
]


def spectral_anomaly(x, q=3):
    """Spectral residual saliency of a one-dimensional record.

    NOT IN SCHABENBERGER & GOTWAY -- this is a signal/vision method, not
    spatial statistics. The source is Hou, X. & Zhang, L. (2007),
    "Saliency detection: a spectral residual approach", CVPR -- named from
    the general literature and NOT verified against a PDF in this corpus.

    The idea: the log-amplitude spectrum of natural signals is smooth and
    largely shared across signals, so what a PARTICULAR signal contributes
    is the residual after that smooth part is averaged away.

        L(w) = log|F(x)(w)|
        R(w) = L(w) - avg_q L(w)                (q-point moving average)
        S    = |IDFT( exp(R(w)) e^{i phase(w)} )|^2

    The PHASE IS KEPT UNCHANGED. Rebuilding from the residual amplitude
    with the original phase is the entire mechanism; discard the phase and
    the reconstruction is noise. The moving average is applied CIRCULARLY,
    matching the periodicity of the DFT.

    A zero amplitude at some frequency makes log undefined; a floor of
    1e-300 is applied and the count of floored ordinates is returned as
    ``floored`` so it is visible rather than silent.

    Parameters
    ----------
    x : (n,) array-like
        Record.
    q : int
        Moving-average width for the log spectrum; odd, at least 1.

    Returns
    -------
    RichResult
        ``saliency``, ``peak``, ``peak_index``, ``residual``,
        ``floored``, ``n``, ``method``.
    """
    v = vec(x, "x")
    n = len(v)
    if n < 8:
        raise ValueError("at least 8 samples are needed")
    q = int(q)
    if q < 1 or q % 2 == 0:
        raise ValueError("`q` must be an odd positive integer")
    if q > n:
        raise ValueError("`q` (%d) exceeds the record length (%d)" % (q, n))

    re, im = dft(v)
    amp = [sqrt(re[k] * re[k] + im[k] * im[k]) for k in range(n)]
    floored = 0
    lg = []
    for a in amp:
        if a < 1e-300:
            floored = floored + 1
            lg.append(log(1e-300))
        else:
            lg.append(log(a))
    half = q // 2
    sm = [fsum([lg[(k + t) % n] for t in range(-half, half + 1)]) / q
          for k in range(n)]
    res = [lg[k] - sm[k] for k in range(n)]
    ph = phase(re, im)
    from math import cos, exp, sin
    nre = [exp(res[k]) * cos(ph[k]) for k in range(n)]
    nim = [exp(res[k]) * sin(ph[k]) for k in range(n)]
    rec = idftre(nre, nim)
    sal = [t * t for t in rec]
    pk = max(sal)
    pi_ = sal.index(pk)

    return RichResult(payload={
        "saliency": sal,
        "peak": pk,
        "peak_index": float(pi_),
        "residual": res,
        "log_amplitude": lg,
        "floored": float(floored),
        "phase_is_preserved": True,
        "q": float(q),
        "n": n,
        "method": ("Spectral residual saliency (Hou & Zhang 2007); NOT in "
                   "Schabenberger & Gotway"),
    })


def cheatsheet():
    return "specA: spectral residual saliency"


# compact alias per ledger/NAMING.md
specanom = spectral_anomaly
