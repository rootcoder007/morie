# morie.fn -- function file (rootcoder007/morie)
"""Decision and factoring algorithms: a lazy DPLL(T) solver for integer difference logic and a state-vector
simulation of Shor's order-finding factorisation."""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._rng import random_uniform
from .satDP import dpll

__all__ = ["smt_solver", "shor_factoring"]


def _neg_cycle(nodes, edges):
    """Bellman-Ford from a virtual source; returns (dist, cycle_edge_indices or None)."""
    dist = {v: 0 for v in nodes}
    pred = {v: None for v in nodes}
    last = None
    for _ in range(len(nodes)):
        last = None
        for k, (u, v, w, _lit) in enumerate(edges):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = k
                last = v
        if last is None:
            return dist, None
    v = last
    for _ in range(len(nodes)):
        v = edges[pred[v]][0]
    cyc, u = [], v
    while True:
        k = pred[u]
        cyc.append(k)
        u = edges[k][0]
        if u == v:
            break
    return dist, cyc[::-1]


def smt_solver(formula, *, max_iter: int = 10000) -> RichResult:
    r"""Lazy DPLL(T) satisfiability for integer difference logic (QF_IDL).

    ``formula = {"atoms": [(x, y, c), ...], "clauses": [[lit, ...], ...]}``:
    atom ``k`` (1-based) is the constraint ``x - y <= c`` over integer
    variables named ``x`` and ``y``; literals are signed atom indices in
    DIMACS style, and indices above ``len(atoms)`` are plain Boolean
    variables. The Boolean skeleton is solved by DPLL (``morie.fn.satDP``);
    the atoms it sets true, and the integer negations ``y - x <= -c - 1`` of
    those it sets false, become edges ``y -> x`` of weight ``c`` in a
    constraint graph. Bellman-Ford either returns potentials (a model:
    ``solution[x] = dist(x)``) or a negative cycle, whose literals are
    conjunctively inconsistent; their negation is learned as a clause and
    the SAT search repeats (Nieuwenhuis, Oliveras and Tinelli 2006).

    References
    ----------
    Nieuwenhuis, R., Oliveras, A. and Tinelli, C. (2006). Solving SAT and
    SAT modulo theories: from an abstract Davis-Putnam-Logemann-Loveland
    procedure to DPLL(T). *Journal of the ACM*, 53, 937-977.
    Cotton, S. and Maler, O. (2006). Fast and flexible difference
    constraint propagation for DPLL(T). *SAT 2006*, LNCS 4121, 170-183.

    Examples
    --------
    >>> f = {"atoms": [("x", "y", -1), ("y", "z", -1), ("z", "x", -1), ("z", "x", 5)],
    ...      "clauses": [[1], [2], [3, 4]]}
    >>> r = smt_solver(f)
    >>> r.satisfiable, r.model, r.solution, r.theory_conflicts
    (True, {1: True, 2: True, 3: False, 4: True}, {'x': -2, 'y': -1, 'z': 0}, 1)
    """
    atoms = [(str(a[0]), str(a[1]), int(a[2])) for a in formula["atoms"]]
    clauses = [[int(v) for v in cl] for cl in formula["clauses"]]
    names = sorted({a[0] for a in atoms} | {a[1] for a in atoms})
    learned = []
    for it in range(int(max_iter)):
        sat = dpll(clauses + learned)
        if not sat.satisfiable:
            return RichResult(
                payload={"satisfiable": False, "model": {}, "solution": {}, "theory_conflicts": it, "learned": learned}
            )
        model = {int(k): bool(v) for k, v in sat.model.items()}
        edges = []
        for k in sorted(model):
            if k > len(atoms):
                continue
            x, y, c = atoms[k - 1]
            if model[k]:
                edges.append((y, x, c, k))
            else:
                edges.append((x, y, -c - 1, -k))
        dist, cyc = _neg_cycle(names, edges)
        if cyc is None:
            return RichResult(
                payload={
                    "satisfiable": True,
                    "model": model,
                    "solution": {v: dist[v] for v in names},
                    "theory_conflicts": it,
                    "learned": learned,
                }
            )
        learned.append(sorted({-edges[k][3] for k in cyc}, key=lambda v: (abs(v), v)))
    raise RuntimeError("smt_solver: max_iter reached")


def _is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _perfect_power(n):
    for k in range(2, n.bit_length() + 1):
        b = round(n ** (1.0 / k))
        for c in (b - 1, b, b + 1):
            if c > 1 and c**k == n:
                return c
    return None


def _convergents(p, q):
    h0, h1, k0, k1 = 0, 1, 1, 0
    out = []
    while q:
        t = p // q
        h0, h1 = h1, t * h1 + h0
        k0, k1 = k1, t * k1 + k0
        out.append((h1, k1))
        p, q = q, p - t * q
    return out


def _order_distribution(r, Q):
    M, rem = Q // r, Q % r
    probs = []
    for y in range(Q):
        ry = (r * y) % Q
        if ry == 0:
            g1, g0 = float((M + 1) ** 2), float(M * M)
        else:
            th = math.pi * ry / Q
            s = math.sin(th)
            g1 = (math.sin((M + 1) * th) / s) ** 2
            g0 = (math.sin(M * th) / s) ** 2
        probs.append((rem * g1 + (r - rem) * g0) / (Q * Q))
    return probs


def shor_factoring(N, *, seed: int = 1, max_attempts: int = 20) -> RichResult:
    r"""Shor's algorithm with the quantum order-finding step simulated from its exact output distribution.

    Classical reductions first: even ``N`` gives 2, a perfect power
    ``b^k`` gives ``b``, a prime ``N`` is rejected. Then for a base ``a``
    drawn uniformly from ``2..N-1`` (Philox) with ``gcd(a, N) = 1``, the
    first register of ``t`` qubits (``Q = 2^t``, ``N^2 <= Q < 2N^2``) is
    measured after the inverse QFT. For ``f(x) = a^x mod N`` of period
    ``r`` the preimage of each second-register value is an arithmetic
    progression of step ``r``, so the outcome ``y`` has probability
    ``Q^-2 sum_c |sum_j exp(2 pi i (x_c + j r) y / Q)|^2`` (Nielsen and
    Chuang 2010, 5.3.1), evaluated here in closed form. The continued
    fraction of ``y / Q`` gives candidate orders (convergent denominators
    below ``N`` with ``a^q = 1 mod N``); an even order with ``a^(r/2) !=
    -1 mod N`` yields the factors ``gcd(a^(r/2) +- 1, N)``. Returns
    ``factors``, ``order``, ``base``, ``Q`` and the attempt log. ``Q`` is
    capped at ``2^20`` (``N < 1024``).

    References
    ----------
    Shor, P. W. (1997). Polynomial-time algorithms for prime factorization
    and discrete logarithms on a quantum computer. *SIAM Journal on
    Computing*, 26, 1484-1509.
    Nielsen, M. A. and Chuang, I. L. (2010). *Quantum Computation and
    Quantum Information*, 10th anniversary edn, section 5.3. Cambridge
    University Press.

    Examples
    --------
    >>> r = shor_factoring(15, seed=2)
    >>> r.factors, r.base, r.order
    ([3, 5], 7, 4)
    """
    N = int(N)
    if N < 4:
        raise ValueError("N must be a composite integer >= 4")
    if N % 2 == 0:
        return RichResult(
            payload={"factors": [2, N // 2], "order": None, "base": None, "Q": None, "attempts": [], "route": "even"}
        )
    b = _perfect_power(N)
    if b is not None:
        return RichResult(
            payload={
                "factors": [b, N // b],
                "order": None,
                "base": None,
                "Q": None,
                "attempts": [],
                "route": "perfect power",
            }
        )
    if _is_prime(N):
        raise ValueError("N is prime")
    t = (N * N - 1).bit_length()
    Q = 2**t
    if Q > 2**20:
        raise ValueError("N too large for the state-vector simulation (Q > 2^20)")
    u = [float(v) for v in random_uniform(2 * int(max_attempts), seed=seed)]
    attempts = []
    for k in range(int(max_attempts)):
        a = 2 + min(int(u[2 * k] * (N - 2)), N - 3)
        g = math.gcd(a, N)
        if g > 1:
            attempts.append({"a": a, "y": None, "r": None, "outcome": "gcd"})
            return RichResult(
                payload={
                    "factors": sorted([g, N // g]),
                    "order": None,
                    "base": a,
                    "Q": Q,
                    "attempts": attempts,
                    "route": "gcd",
                }
            )
        r = 1
        v = a % N
        while v != 1:
            v = v * a % N
            r += 1
        probs = _order_distribution(r, Q)
        y, acc = Q - 1, 0.0
        for i, p in enumerate(probs):
            acc += p
            if u[2 * k + 1] < acc:
                y = i
                break
        rr = None
        for _, q in _convergents(y, Q):
            if 0 < q < N and pow(a, q, N) == 1:
                rr = q
                break
        if rr is None:
            attempts.append({"a": a, "y": y, "r": None, "outcome": "no order from y/Q"})
            continue
        if rr % 2 or pow(a, rr // 2, N) == N - 1:
            attempts.append({"a": a, "y": y, "r": rr, "outcome": "odd order or a^(r/2) = -1"})
            continue
        h = pow(a, rr // 2, N)
        f = math.gcd(h - 1, N)
        if f in (1, N):
            f = math.gcd(h + 1, N)
        attempts.append({"a": a, "y": y, "r": rr, "outcome": "factored"})
        return RichResult(
            payload={
                "factors": sorted([f, N // f]),
                "order": rr,
                "base": a,
                "Q": Q,
                "attempts": attempts,
                "route": "order finding",
            }
        )
    raise RuntimeError("shor_factoring: no factor found within max_attempts")


def cheatsheet() -> str:
    return "smt_solver(formula) -> lazy DPLL(T) for difference logic; shor_factoring(N) -> simulated order finding."
