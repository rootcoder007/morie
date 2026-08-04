# morie.fn -- function file (rootcoder007/morie)
"""Structured state-space convolution kernel."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["ssmk", "s4ssmkernel", "s4_ssm_kernel"]


def ssmk(A, B, C, L):
    """Structured state-space convolution kernel.

    State-space kernel: K_l = C A^l B, y = K * x.

    Gu, Goel & Re (2022), S4.  A linear state-space model unrolled in
    time is a convolution with the kernel (CB, CAB, CA^2 B, ...), so a
    recurrence of length L becomes one convolution -- that equivalence
    is what makes the model trainable at long sequence lengths.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Structured state-space convolution kernel", payload=_c.ssmk(A=A, B=B, C=C, L=L))


s4_ssm_kernel = ssmk


def cheatsheet():
    return "ssmkrn: Structured state-space convolution kernel"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
s4ssmkernel = ssmk
