# morie.fn -- function file (rootcoder007/morie)
r"""The parallel scan that makes a state space model trainable.

A linear recurrence :math:`x_t = A_t x_{t-1} + b_t` is sequential by
definition -- :math:`L` steps, no parallelism, which is exactly why
RNNs lost to Transformers on modern hardware. The escape is that this
recurrence is an **associative** operation in disguise.

**Write each step as an affine map and compose them.** With
:math:`(A_2, b_2)\circ(A_1, b_1) = (A_2A_1,\; A_2b_1 + b_2)`,
composition is associative, so the prefix products can be computed by
a **parallel scan**: an up-sweep building partial compositions and a
down-sweep distributing them, :math:`O(L)` work in
:math:`O(\log L)` depth. ``parallel_scan`` returns the same sequence
as ``sequential_scan`` -- exactly, to floating-point, which the anchor
checks -- but the *depth* is logarithmic.

**Associativity is the load-bearing property**, so ``check_
associativity`` tests it directly rather than trusting the algebra: if
the composition were not associative, the scan would silently return a
different answer depending on how the tree was cut.

**Why this is the thing that made selective SSMs viable.** Once the
recurrence's parameters are allowed to depend on the input -- Mamba's
selection mechanism -- the convolutional shortcut used by earlier SSMs
disappears, because there is no longer a single fixed kernel. What
remains is this scan, which is why the hardware-aware parallel scan is
presented as the enabling implementation rather than an optimisation.

References
----------
Smith, J. T. H., Warrington, A. & Linderman, S. W. (2023)
"Simplified State Space Layers for Sequence Modeling",
*International Conference on Learning Representations (ICLR 2023)*,
arXiv:2208.04933. The use of a parallel associative scan over the
linear state space recurrence, with the binary operator composing
affine maps, giving O(L) work and O(log L) depth and removing the need
for the convolutional/kernel formulation.

Gu, A. & Dao, T. (2024) "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces", *Conference on Language Modeling (COLM
2024)*, arXiv:2312.00752. That making the SSM parameters functions of
the input costs the efficient convolution, and that a hardware-aware
parallel algorithm in recurrent mode is what recovers the speed.

Blelloch, G. E. (1990) "Prefix Sums and Their Applications",
Technical Report CMU-CS-90-190, School of Computer Science, Carnegie
Mellon University. The up-sweep/down-sweep work-efficient scan.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["compose", "sequential_scan", "parallel_scan",
           "check_associativity", "scan_depth"]

_EPS = 1e-12


def compose(left, right):
    r""":math:`(A_2,b_2)\circ(A_1,b_1) = (A_2A_1, A_2b_1+b_2)`.

    ``left`` is applied first. Associative, which is the entire
    reason the recurrence parallelises.
    """
    A1, b1 = float(left[0]), float(left[1])
    A2, b2 = float(right[0]), float(right[1])
    return (A2 * A1, A2 * b1 + b2)


def sequential_scan(pairs, x0=0.0):
    r"""The recurrence as written: :math:`L` steps, no parallelism."""
    P = [(float(a), float(b)) for a, b in pairs]
    x = float(x0)
    out = []
    for (A, b) in P:
        x = A * x + b
        out.append(x)
    return {"states": out, "steps": len(P), "depth": len(P),
            "note": "depth equals length -- the reason RNNs do not "
                    "use the hardware"}


def _upsweep(P):
    tree, level = [], list(P)
    while len(level) > 1:
        tree.append(level)
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(compose(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    tree.append(level)
    return tree


def parallel_scan(pairs, x0=0.0):
    r"""Blelloch scan over the affine composition.

    Same states as :func:`sequential_scan`, :math:`O(\log L)` depth.
    """
    P = [(float(a), float(b)) for a, b in pairs]
    n = len(P)
    if n == 0:
        raise ValueError("ssmpar: the sequence is empty")
    tree = _upsweep(P)
    prefix = [None] * n
    for i in range(n):
        acc = P[i]
        j = i - 1
        step = 1
        while j >= 0:
            lo = j - step + 1
            if lo >= 0 and (j + 1) % step == 0 and step <= j + 1:
                blk = P[lo]
                for t in range(lo + 1, j + 1):
                    blk = compose(blk, P[t])
                acc = compose(blk, acc)
                j = lo - 1
                step *= 2
            else:
                acc = compose(P[j], acc)
                j -= 1
        prefix[i] = acc
    x = float(x0)
    states = [A * x + b for (A, b) in prefix]
    return {"states": states, "prefix": prefix,
            "depth": len(tree) - 1 if len(tree) > 1 else 1,
            "work": n,
            "note": "identical states, logarithmic DEPTH"}


def check_associativity(a, b, c, tol=1e-12):
    r"""Test :math:`(a\circ b)\circ c = a\circ(b\circ c)` directly.

    If this failed, the scan would give a different answer depending
    on where the tree was cut -- so it is checked, not assumed.
    """
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    d = max(abs(left[0] - right[0]), abs(left[1] - right[1]))
    return {"left": left, "right": right, "deviation": d,
            "associative": d <= float(tol),
            "note": "the property the parallel scan rests on"}


def scan_depth(length):
    r"""Sequential against parallel depth."""
    n = int(length)
    if n < 1:
        raise ValueError("ssmpar: the length must be positive")
    d = max(1, int(math.ceil(math.log(n, 2)))) if n > 1 else 1
    return RichResult(payload={
        "estimate": d, "parallel_depth": d, "sequential_depth": n,
        "work": n, "speedup": n / float(d),
        "method": "parallel associative scan; Smith, Warrington & "
                  "Linderman (2023), after Blelloch (1990)",
        "note": "O(L) work in O(log L) depth; with input-dependent "
                "parameters (Mamba) the convolutional shortcut is "
                "gone and this scan is what is left",
    })


def cheatsheet():
    return ("ssmpar: x_t = A_t x_{t-1} + b_t looks sequential, but "
            "each step is an AFFINE MAP and composition "
            "(A2,b2)o(A1,b1) = (A2A1, A2b1+b2) is ASSOCIATIVE -- so the "
            "prefixes come from a parallel scan: O(L) work, O(log L) "
            "depth, identical states. Associativity is load-bearing "
            "(cut the tree anywhere and the answer must not change), "
            "so test it. This is what makes SELECTIVE state space "
            "models viable: once the parameters depend on the input "
            "there is no fixed convolution kernel left, and the scan "
            "is the only route to the hardware.")


# compact alias per ledger/NAMING.md
parallelscan = parallel_scan
