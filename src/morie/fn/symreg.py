# morie.fn -- function file (rootcoder007/morie)
"""Symbolic regression in the PySR style: regularized evolution of expression trees with a complexity-indexed
hall of fame, its Pareto front, and PySR's score-based model selection."""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["pysr_regression"]

_BIN = ("+", "-", "*", "/")
_UN = ("sin", "cos", "exp", "log", "sqrt", "square")


class _U:
    def __init__(self, seed):
        self.seed, self.block, self.buf, self.pos = seed, 0, [], 0

    def __call__(self):
        if self.pos >= len(self.buf):
            self.buf = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def _arity(node):
    return 2 if node[0] == 2 else (1 if node[0] == 3 else 0)


def _span(t, i):
    need, j = 1, i
    while need > 0:
        need += _arity(t[j]) - 1
        j += 1
    return j


def _div(a, b):
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _un(name, a):
    try:
        if name == "sin":
            return math.sin(a)
        if name == "cos":
            return math.cos(a)
        if name == "exp":
            return math.exp(a)
        if name == "log":
            if a == 0:
                return -math.inf
            return math.log(a) if a > 0 else math.nan
        if name == "sqrt":
            return math.sqrt(a) if a >= 0 else math.nan
        return a * a
    except OverflowError:
        return math.inf
    except ValueError:
        return math.nan


def _eval(t, X, un):
    pos = [0]

    def rec():
        k, v = t[pos[0]]
        pos[0] += 1
        if k == 0:
            return [v] * len(X)
        if k == 1:
            return [row[v] for row in X]
        if k == 3:
            a = rec()
            return [_un(un[v], x) for x in a]
        a = rec()
        b = rec()
        op = _BIN[v]
        if op == "+":
            return [x + y for x, y in zip(a, b)]
        if op == "-":
            return [x - y for x, y in zip(a, b)]
        if op == "*":
            return [x * y for x, y in zip(a, b)]
        return [_div(x, y) for x, y in zip(a, b)]

    return rec()


def _loss(t, X, y, un):
    pred = _eval(t, X, un)
    s = 0.0
    for p, v in zip(pred, y):
        d = p - v
        s += d * d
    s /= len(y)
    return s if math.isfinite(s) else math.inf


def _grow(U, d, p, un):
    if d <= 0 or U() < 0.3:
        return _terminal(U, p)
    if un and U() < 0.2:
        return [(3, min(int(U() * len(un)), len(un) - 1))] + _grow(U, d - 1, p, un)
    op = (2, min(int(U() * 4), 3))
    left = _grow(U, d - 1, p, un)
    return [op] + left + _grow(U, d - 1, p, un)


def _terminal(U, p):
    if U() < 0.4:
        return [(0, 4.0 * U() - 2.0)]
    return [(1, min(int(U() * p), p - 1))]


def _mutate(U, t, p, un):
    u = U()
    kinds = {0: [], 1: [], 2: [], 3: []}
    for i, (k, _v) in enumerate(t):
        kinds[k].append(i)
    if u < 0.3 and kinds[0]:
        i = kinds[0][min(int(U() * len(kinds[0])), len(kinds[0]) - 1)]
        out = list(t)
        out[i] = (0, t[i][1] + 2.0 * (U() - 0.5))
        return out
    ops = sorted(kinds[2] + kinds[3])
    if 0.3 <= u < 0.5 and ops:
        i = ops[min(int(U() * len(ops)), len(ops) - 1)]
        out = list(t)
        n = 4 if t[i][0] == 2 else len(un)
        out[i] = (t[i][0], min(int(U() * n), n - 1))
        return out
    if 0.5 <= u < 0.65 and kinds[1]:
        i = kinds[1][min(int(U() * len(kinds[1])), len(kinds[1]) - 1)]
        out = list(t)
        out[i] = (1, min(int(U() * p), p - 1))
        return out
    i = min(int(U() * len(t)), len(t) - 1)
    j = _span(t, i)
    if u >= 0.85:
        op = (2, min(int(U() * 4), 3))
        return t[:i] + [op] + t[i:j] + _terminal(U, p) + t[j:]
    return t[:i] + _grow(U, 2, p, un) + t[j:]


def _crossover(U, a, b):
    i = min(int(U() * len(a)), len(a) - 1)
    j = min(int(U() * len(b)), len(b) - 1)
    return a[:i] + b[j : _span(b, j)] + a[_span(a, i) :]


def _string(t, un, names):
    pos = [0]

    def rec():
        k, v = t[pos[0]]
        pos[0] += 1
        if k == 0:
            return f"{v:.6g}"
        if k == 1:
            return names[v]
        if k == 3:
            return f"{un[v]}({rec()})"
        a = rec()
        b = rec()
        return f"({a} {_BIN[v]} {b})"

    return rec()


def pysr_regression(
    X,
    y,
    *,
    unary_operators=(),
    niterations: int = 30,
    population_size: int = 60,
    maxsize: int = 20,
    tournament_size: int = 8,
    crossover_probability: float = 0.1,
    parsimony: float = 0.0032,
    seed: int = 1,
) -> RichResult:
    r"""Symbolic regression by regularized evolution with a Pareto front over complexity (PySR's scheme).

    Expressions are prefix trees over ``+ - * /``, the chosen unary
    operators (``sin cos exp log sqrt square``), the variables ``x0, x1,
    ...`` and real constants. Each of ``niterations x population_size``
    steps picks a parent by tournament on ``loss + parsimony x size``
    (``loss`` the mean squared error), makes a child by subtree crossover
    with a second winner (probability ``crossover_probability``) or by one
    mutation (constant shift, operator swap, variable swap, subtree
    regrowth, node insertion), rejects children larger than ``maxsize``,
    and replaces the oldest member (regularized, age-based evolution). The
    hall of fame keeps the lowest-loss expression of every size; its
    Pareto front lists them with PySR's score ``-d log(loss) / d size``,
    and ``best`` is the highest-scoring front member whose loss is within
    1.5 times the smallest (PySR ``model_selection="best"``). Draws come
    from Philox streams, so the R twin evolves the same expressions.
    Constants are evolved, not optimised numerically.

    References
    ----------
    Cranmer, M. (2023). Interpretable machine learning for science with
    PySR and SymbolicRegression.jl. arXiv:2305.01582.
    Real, E., Aggarwal, A., Huang, Y. and Le, Q. V. (2019). Regularized
    evolution for image classifier architecture search. *AAAI*, 33,
    4780-4789.
    Koza, J. R. (1992). *Genetic Programming*. MIT Press.

    Examples
    --------
    >>> X = [[i / 4.0] for i in range(-8, 9)]
    >>> y = [2.0 * r[0] * r[0] + r[0] for r in X]
    >>> r = pysr_regression(X, y, niterations=40, seed=3)
    >>> r.best["loss"] < 1e-20, r.best["equation"]
    (True, '((x0 * (x0 + x0)) + x0)')
    """
    Xm = [[float(v) for v in row] for row in X]
    yv = [float(v) for v in y]
    if len(Xm) != len(yv) or not Xm:
        raise ValueError("X and y must be non-empty and of equal length")
    p = len(Xm[0])
    un = tuple(unary_operators)
    for o in un:
        if o not in _UN:
            raise ValueError(f"unknown unary operator {o!r}")
    names = [f"x{j}" for j in range(p)]
    U = _U(seed)
    P = int(population_size)
    pop, losses = [], []
    for _ in range(P):
        t = _grow(U, 1 + min(int(U() * 3), 2), p, un)
        if len(t) > maxsize:
            t = _terminal(U, p)
        pop.append(t)
        losses.append(_loss(t, Xm, yv, un))
    hof = {}

    def record(t, loss):
        c = len(t)
        if loss < math.inf and (c not in hof or loss < hof[c][1]):
            hof[c] = (list(t), loss)

    for t, loss in zip(pop, losses):
        record(t, loss)

    def tournament():
        best = -1
        for _ in range(int(tournament_size)):
            i = min(int(U() * P), P - 1)
            if best < 0 or losses[i] + parsimony * len(pop[i]) < losses[best] + parsimony * len(pop[best]):
                best = i
        return best

    oldest = 0
    for _ in range(int(niterations) * P):
        a = tournament()
        if U() < crossover_probability:
            b = tournament()
            child = _crossover(U, pop[a], pop[b])
        else:
            child = _mutate(U, pop[a], p, un)
        if len(child) > maxsize:
            child = list(pop[a])
        cl = _loss(child, Xm, yv, un)
        pop[oldest], losses[oldest] = child, cl
        oldest = (oldest + 1) % P
        record(child, cl)
    front = []
    for c in sorted(hof):
        t, loss = hof[c]
        if not front or loss < front[-1]["loss"]:
            front.append({"complexity": c, "loss": loss, "equation": _string(t, un, names), "tree": t})
    for k, e in enumerate(front):
        if k == 0:
            e["score"] = 0.0
        else:
            prev = front[k - 1]
            lo, lp = max(e["loss"], 1e-300), max(prev["loss"], 1e-300)
            e["score"] = -(math.log(lo) - math.log(lp)) / (e["complexity"] - prev["complexity"])
    lmin = min(e["loss"] for e in front)
    ok = [e for e in front if e["loss"] <= 1.5 * lmin]
    best = max(ok, key=lambda e: (e["score"], -e["complexity"]))
    pred = _eval(best["tree"], Xm, un)
    return RichResult(
        payload={
            "equations": [{k: v for k, v in e.items() if k != "tree"} for e in front],
            "best": {k: v for k, v in best.items() if k != "tree"},
            "prediction": pred,
        }
    )


def cheatsheet() -> str:
    return "pysr_regression(X, y) -> Pareto front of evolved expressions and PySR's 'best' model."
