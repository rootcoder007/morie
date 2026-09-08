# morie.fn -- function file (rootcoder007/morie)
"""Barber-Agakov variational lower bound on mutual information.

SOURCE.  Barber, D. and Agakov, F. (2003), "The IM Algorithm: A
Variational Approach to Information Maximization", *Advances in Neural
Information Processing Systems* 16 (NIPS 2003), MIT Press.

The bound is obtained by writing the mutual information as
I(X;Y) = H(X) - H(X|Y) and replacing the intractable posterior p(x|y)
by any variational decoder q(x|y).  Because

    H(X|Y) = -E_{p(x,y)}[log p(x|y)]
           = -E_{p(x,y)}[log q(x|y)] - E_{p(y)}[ KL( p(.|y) || q(.|y) ) ]

and the KL term is non-negative, dropping it can only lower the value:

    I(X;Y) >= H(X) + E_{p(x,y)}[ log q(x|y) ]                     (BA bound)

with equality exactly when q(x|y) = p(x|y) for every y that occurs.
The gap is the average KL, so ``gap`` in the payload is a non-negative
quantity by Gibbs' inequality -- a property this module asserts rather
than assumes.

SCOPE.  Discrete X and Y, with the joint estimated by counting.  All
quantities are in nats.  ``q`` may be supplied as a matrix whose row j
is the decoder distribution q(. | y = level j) over the levels of X; if
it is omitted the empirical conditional is used, which makes the bound
tight and equal to the plug-in mutual information.  A continuous
version would need a parametric decoder and an optimiser; that is not
implemented, and the omission is this implementation's scope choice.

The equivalent form quoted by the stub docstring,
I(X;Y) >= E[log q(y|x)/p(y)], is the same bound with the roles of X and
Y exchanged; it is obtained here by calling the function with the
arguments swapped.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["variational_bound"]


def _levels(v):
    seen = []
    for e in v:
        if e not in seen:
            seen.append(e)
    seen.sort()
    return seen


def variational_bound(X, Y, q=None):
    """Barber-Agakov lower bound on I(X;Y) for discrete variables.

    Parameters
    ----------
    X, Y : sequence
        Paired discrete observations of equal length.
    q : array-like or None
        ``ny``-by-``nx`` matrix; row j is q(. | y = level j).  Rows must
        be non-negative and sum to one.  ``None`` uses the empirical
        conditional.

    Returns
    -------
    RichResult
        ``bound``, ``entropy_x``, ``entropy_y``, ``expected_log_q``,
        ``plugin_mi``, ``gap``, ``conditional_entropy``, ``n``, ``nx``,
        ``ny``.

    Raises
    ------
    ValueError
        Empty or mismatched inputs, a ``q`` of the wrong shape, a row of
        ``q`` that is not a distribution, or a zero decoder probability
        on an observed pair (the bound is then -infinity).

    References
    ----------
    Barber, D. and Agakov, F. (2003).  Advances in Neural Information
    Processing Systems 16 (NIPS 2003).  MIT Press.
    """
    xs = list(X)
    ys = list(Y)
    n = len(xs)
    if n == 0:
        raise ValueError("variational_bound: X is empty")
    if len(ys) != n:
        raise ValueError("variational_bound: X and Y must have the same length")
    lx = _levels(xs)
    ly = _levels(ys)
    nx = len(lx)
    ny = len(ly)
    ix = {v: i for i, v in enumerate(lx)}
    iy = {v: i for i, v in enumerate(ly)}
    joint = [[0.0] * nx for _ in range(ny)]
    for a, b in zip(xs, ys):
        joint[iy[b]][ix[a]] += 1.0
    px = [0.0] * nx
    py = [0.0] * ny
    for j in range(ny):
        for i in range(nx):
            joint[j][i] /= n
            px[i] += joint[j][i]
            py[j] += joint[j][i]
    if q is None:
        Q = [[(joint[j][i] / py[j] if py[j] > 0.0 else 0.0) for i in range(nx)]
             for j in range(ny)]
    else:
        Q = core.mat(q)
        if len(Q) != ny or any(len(r) != nx for r in Q):
            raise ValueError("variational_bound: q must be ny-by-nx")
        for j in range(ny):
            s = 0.0
            for i in range(nx):
                if Q[j][i] < 0.0:
                    raise ValueError("variational_bound: q has a negative entry")
                s += Q[j][i]
            if abs(s - 1.0) > 1e-9:
                raise ValueError("variational_bound: each row of q must sum to one")
    hx = 0.0
    for i in range(nx):
        if px[i] > 0.0:
            hx -= px[i] * math.log(px[i])
    hy = 0.0
    for j in range(ny):
        if py[j] > 0.0:
            hy -= py[j] * math.log(py[j])
    elq = 0.0
    for j in range(ny):
        for i in range(nx):
            if joint[j][i] > 0.0:
                if not (Q[j][i] > 0.0):
                    raise ValueError(
                        "variational_bound: q assigns zero probability to an observed pair")
                elq += joint[j][i] * math.log(Q[j][i])
    hxy = 0.0
    for j in range(ny):
        for i in range(nx):
            if joint[j][i] > 0.0 and py[j] > 0.0:
                hxy -= joint[j][i] * math.log(joint[j][i] / py[j])
    mi = hx - hxy
    bound = hx + elq
    return RichResult(
        title="Barber-Agakov variational bound on mutual information",
        summary_lines=[("pairs", n), ("bound (nats)", bound), ("plug-in MI", mi)],
        payload={
            "estimate": bound,
            "bound": bound,
            "entropy_x": hx,
            "entropy_y": hy,
            "expected_log_q": elq,
            "plugin_mi": mi,
            "gap": mi - bound,
            "conditional_entropy": hxy,
            "n": n,
            "nx": nx,
            "ny": ny,
            "method": "I(X;Y) >= H(X) + E[log q(x|y)] (Barber and Agakov 2003)",
        },
    )


def cheatsheet():
    return "vbinfp: Barber-Agakov variational lower bound on mutual information"
