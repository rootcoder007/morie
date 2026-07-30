# morie.fn -- function file (rootcoder007/morie)
"""Strict convexity test -- Boyd & Vandenberghe Sec. 3.1.1 / 9.1.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_strict_convex"]


def boyd_strict_convex(f, lower=-1.0, upper=1.0, n_dim=1, n_pairs=400,
                       tol=1e-09, seed=0):
    r"""Probe whether :math:`f` is convex, STRICTLY convex, or neither.

    Convexity is :math:`f((1-t)x + ty) \le (1-t)f(x) + tf(y)`; strict
    convexity replaces :math:`\le` with :math:`<` for :math:`x \ne y`
    and :math:`t \in (0,1)`. The gap between them is not pedantic --
    it is the difference between a set of minimisers and a single point.
    A convex function can be flat along a whole face (any linear
    function is convex, and :math:`\lVert x\rVert_1` is flat along the
    axes), so its minimiser need not be unique; a strictly convex one
    has at most one.

    A third rung above: :math:`f` is STRONGLY convex with modulus
    :math:`m` when :math:`f - \tfrac{m}{2}\lVert x\rVert^2` is still
    convex, which forces a quadratic lower bound and hence linear
    convergence for gradient descent. Strict convexity alone buys
    uniqueness but no rate -- :math:`x^4` is strictly convex with
    modulus 0 at the origin. All three are reported separately because
    they underwrite different guarantees.

    The modulus is estimated from the midpoint deficit, which for a
    strongly convex :math:`f` satisfies

    .. math::

        \tfrac12(f(x)+f(y)) - f\!\left(\tfrac{x+y}{2}\right)
        \ge \tfrac{m}{8}\lVert x-y\rVert^2 .

    ponytail: this SAMPLES chords, so it can refute convexity outright
    but can only fail to find a counterexample in the other direction.
    Read ``convex=True`` as "no violation found on this domain", not as
    a proof.

    Parameters
    ----------
    f : callable
        ``f(x) -> float`` with ``x`` a 1-d array.
    lower, upper : float or array-like
        Box the chords are sampled from.
    n_dim : int
        Dimension of the domain.
    n_pairs : int
        Number of random chords tested.
    tol : float
        Absolute slack allowed before a violation counts.
    seed : int
        RNG seed.

    Returns
    -------
    RichResult
        ``convex``, ``strictly_convex``, ``strongly_convex``,
        ``modulus`` (estimated ``m``), ``worst_violation``,
        ``min_deficit``, ``unique_minimiser`` (implied by strictness),
        ``n_pairs``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A quadratic is strictly AND strongly convex, and the midpoint
    identity is exact for it, so the estimated modulus is the second
    derivative on the nose.

    >>> sq = boyd_strict_convex(lambda x: x[0] ** 2)
    >>> bool(sq["convex"]), bool(sq["strictly_convex"])
    (True, True)
    >>> round(float(sq["modulus"]), 9)
    2.0

    The absolute value is convex but NOT strictly so: it is linear on
    each half, so any chord that stays on one side achieves equality.
    Its minimiser happens to be unique, but nothing here guarantees it.

    >>> av = boyd_strict_convex(lambda x: abs(x[0]))
    >>> bool(av["convex"]), bool(av["strictly_convex"])
    (True, False)
    >>> round(float(av["modulus"]), 9)
    0.0

    A linear function is the extreme case -- convex, concave, and flat
    everywhere, so every chord holds with equality.

    >>> lin = boyd_strict_convex(lambda x: 3.0 * x[0] - 1.0)
    >>> bool(lin["convex"]), bool(lin["strictly_convex"])
    (True, False)

    ``x^4`` separates the last two rungs. It is strictly convex, so its
    minimiser is unique -- but its curvature vanishes at the origin, so
    it is not STRONGLY convex and gradient descent on it does not
    converge linearly. Sampling cannot refute strong convexity outright
    any more than it can prove convexity, so the evidence is the way the
    modulus COLLAPSES as the domain closes on the origin: shrinking the
    box tenfold divides it by a hundred, the exact ``domain^2`` scaling
    of a curvature that is going to zero.

    >>> q4 = boyd_strict_convex(lambda x: x[0] ** 4)
    >>> bool(q4["strictly_convex"])
    True
    >>> near = boyd_strict_convex(lambda x: x[0] ** 4, -0.1, 0.1)
    >>> round(float(q4["modulus"] / near["modulus"]), 3)
    100.0

    A quadratic's modulus does NOT move when the domain shrinks -- that
    is what being strongly convex means, and the contrast is the test.

    >>> tight = boyd_strict_convex(lambda x: x[0] ** 2, -0.1, 0.1)
    >>> round(float(tight["modulus"]), 6)
    2.0

    A cubic is refuted outright -- one chord below the function is
    enough, which is the asymmetry of the test: it disproves, it does
    not prove.

    >>> cu = boyd_strict_convex(lambda x: x[0] ** 3)
    >>> bool(cu["convex"]), bool(cu["worst_violation"] > 0)
    (False, True)

    In two dimensions the modulus is bounded by the SMALLEST curvature
    direction, not the average: for ``x1^2 + 9*x2^2`` it approaches 2,
    the flat direction's value, not 10.

    >>> d2 = boyd_strict_convex(lambda x: x[0] ** 2 + 9.0 * x[1] ** 2,
    ...                         n_dim=2, n_pairs=2000)
    >>> bool(2.0 <= d2["modulus"] < 2.6)
    True
    """
    if not callable(f):
        raise TypeError("f must be callable")
    n_dim = int(n_dim)
    if n_dim < 1:
        raise ValueError(f"n_dim must be at least 1, got {n_dim}")
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n_dim,)).astype(float)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n_dim,)).astype(float)
    if np.any(hi <= lo):
        raise ValueError("upper must exceed lower in every coordinate")
    n_pairs = int(n_pairs)
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")

    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, size=(n_pairs, n_dim))
    Y = rng.uniform(lo, hi, size=(n_pairs, n_dim))
    # A random t rather than the midpoint alone: a function can satisfy
    # the midpoint inequality and still fail convexity off-centre
    # (midpoint convexity only implies convexity for CONTINUOUS f, and
    # nothing here checks continuity).
    T = rng.uniform(0.05, 0.95, size=n_pairs)

    fx = np.array([float(f(x)) for x in X])
    fy = np.array([float(f(y)) for y in Y])
    Z = (1.0 - T)[:, None] * X + T[:, None] * Y
    fz = np.array([float(f(z)) for z in Z])
    chord = (1.0 - T) * fx + T * fy
    # deficit > 0 is the strict inequality holding on that chord.
    deficit = chord - fz
    d2 = np.sum((X - Y) ** 2, axis=1)
    # Scale by the LOCAL magnitude of f, not by a constant. A strictly
    # convex function's deficit vanishes with the chord -- for x^4 near
    # the origin it is O(a^4) -- so any absolute floor eventually calls
    # it non-strict. Relative to the function values on the same chord
    # the ratio stays O(1), while a genuine equality (a linear stretch)
    # stays at exactly zero.
    scale = np.maximum(np.abs(fx) + np.abs(fy) + np.abs(fz), 1e-12)
    keep = d2 > 1e-12
    violation = float(np.max(-deficit / scale)) if deficit.size else 0.0
    convex = bool(violation <= tol)
    if np.any(keep):
        min_def = float(np.min(deficit[keep] / scale[keep]))
        # Strong convexity with modulus m gives
        # deficit >= (m/2) t (1-t) |x-y|^2; the midpoint case t = 1/2
        # is the familiar m/8 form.
        mod = float(np.min(2.0 * deficit[keep]
                           / (T[keep] * (1.0 - T[keep]) * d2[keep])))
        mod = max(mod, 0.0)
    else:
        min_def, mod = 0.0, 0.0
    strict = bool(convex and min_def > tol)
    return RichResult(
        title="Convexity probe",
        summary_lines=[("dimension", n_dim), ("chords", n_pairs),
                       ("convex", convex), ("strict", strict),
                       ("modulus", mod)],
        payload={
            "convex": convex, "strictly_convex": strict,
            "strongly_convex": bool(strict and mod > 1e-06),
            "modulus": mod,
            "worst_violation": max(violation, 0.0),
            "min_deficit": min_def,
            "unique_minimiser": strict,
            "n_pairs": n_pairs, "method": "boyd_strict_convex",
        },
    )


def cheatsheet():
    return "cvxstx: convex = minimisers may be a SET; strict = at most one; strong = plus a rate. x^4 is strict, not strong"
