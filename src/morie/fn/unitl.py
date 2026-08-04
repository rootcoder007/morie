# morie.fn -- function file (rootcoder007/morie)
"""Unit-length (L2) normalization of a marker vector.

Searched the MVSML (2022) chapter-2 split PDF: sec. 2.6 "Normalization
Methods" pp.57-58 defines exactly five methods -- centering, scaling,
standardization, max normalization and minimax normalization -- and
unit-length normalization is NOT among them.  No book equation is
claimed here.  Dividing a vector by its Euclidean norm is classical
linear algebra with no single owning source, so the docstring says so
rather than attaching a citation that does not support it.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["unit_length_normalization"]


def unit_length_normalization(x):
    """Scale a vector to unit Euclidean length, x / ||x||_2.

    The result lies on the unit sphere, so only the direction of x
    survives.  A zero vector has no direction and is returned
    unchanged, with ``norm`` reported as 0.

    Parameters
    ----------
    x : array-like, the vector to normalize.

    Returns
    -------
    RichResult with keys estimate (the norm that was divided out),
    x_unit, norm, n, method.

    References
    ----------
    Classical; no single owning source.  MVSML (2022) sec. 2.6
    pp.57-58 lists five normalizations and this is not one of them.
    """
    v = [float(t) for t in x]
    nrm = math.sqrt(sum(t * t for t in v))
    out = [t / nrm for t in v] if nrm > 0 else list(v)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(nrm), "x_unit": out, "norm": float(nrm),
        "n": len(v), "method": "unit-length (L2) normalization",
    }), "unitl")


def cheatsheet():
    return "unitl: Unit-length normalization of marker vectors"


# compact alias per ledger/NAMING.md
unitlen = unit_length_normalization
