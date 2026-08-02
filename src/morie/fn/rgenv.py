# morie.fn -- function file (rootcoder007/morie)
"""Hilbert-transform envelope -- Rangayyan & Krishnan Sec 5.5.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_envelope"]


def rangayyan_envelope(x):
    """Analytic-signal envelope via the Hilbert transform.

    ``env(t) = |x(t) + j H{x(t)}|`` where H{·} is the discrete Hilbert
    transform (``scipy.signal.hilbert``).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    RichResult with keys ``envelope``, ``analytic``,
    ``instantaneous_phase``, ``instantaneous_freq``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.5.3 "The envelogram", p.281
        (Sec 5.5 "Envelope Extraction and Analysis", p.277).
    """
    from ._signal_core import hilbert

    x = np.asarray(x, dtype=float)
    z = hilbert(x)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase) / (2 * np.pi)
    res = RichResult(
        title="Hilbert envelope",
        summary_lines=[
            ("N samples", int(x.size)),
            ("Envelope mean", float(env.mean())),
            ("Envelope max", float(env.max())),
        ],
        interpretation=f"Analytic envelope mean {env.mean():.4g}, peak {env.max():.4g}.",
        payload={"envelope": env, "analytic": z, "instantaneous_phase": phase, "instantaneous_freq": inst_freq},
    )
    return with_describe_pointer(res, "rgenv")


# CANONICAL TEST
# >>> t = np.arange(100)/100.0
# >>> x = np.cos(2*np.pi*5*t) * (1 + 0.3*np.cos(2*np.pi*0.5*t))
# >>> r = rangayyan_envelope(x)
# >>> r["envelope"].shape == x.shape
# True


def cheatsheet():
    return "rgenv: Hilbert envelope -- Rangayyan & Krishnan Sec 5.5.3"
