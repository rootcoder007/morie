"""Genetic programming (Koza 1992).

Programs are expression trees over a function set and a terminal set, and
they are bred rather than written: evaluate every tree against the fitness
cases, select parents in proportion to how well they did, cross them by
swapping subtrees, and repeat. The representation is the point -- because
the genome IS a program, crossover produces something runnable rather than
something that has to be decoded.

Everything here follows the book:

  initialisation   ramped half-and-half. Depths are spread evenly over
                   2..max_depth_init and half of each depth is grown
                   (branches may stop early at a terminal) and half is
                   full (every branch runs to the depth). This is what
                   gives the initial population a spread of shapes
                   instead of one shape repeated.
  fitness          raw fitness is the sum of absolute errors over the
                   fitness cases; standardised fitness equals it here
                   because smaller is better; adjusted fitness is
                   1/(1+standardised), which is what selection uses.
                   Adjusted fitness is not decoration -- it compresses
                   the difference between bad individuals and magnifies
                   the difference between good ones.
  selection        fitness-proportionate over adjusted fitness.
  crossover        pick a node in each parent and swap the subtrees.
                   Koza biases the choice: 90% of the time an internal
                   (function) node, 10% a leaf, because uniform choice
                   picks leaves almost always in a tree of any size and
                   crossover degenerates into swapping constants.
  operators        crossover, reproduction and mutation with the book's
                   probabilities, plus a depth cap on offspring; a child
                   that exceeds it is replaced by the first parent,
                   which is Koza's own rule.

Protected division returns 1.0 for a zero divisor, again from the book:
the alternative is that a single division by zero makes an otherwise good
program worthless.

The random stream is a 32-bit xorshift written out here rather than taken
from either language's generator, because the two do not agree and a run
has to reproduce across both arms. Every draw is exact integer
arithmetic.

Reference
  Koza, J.R. (1992) "Genetic Programming: On the Programming of Computers
    by Means of Natural Selection." MIT Press, Cambridge MA. Chapters 6
    and 7: the tableau, ramped half-and-half, adjusted fitness,
    fitness-proportionate selection, and the 90/10 crossover point bias.
"""

from ._richresult import RichResult

__all__ = ["genetic_programming", "karpV", "cheatsheet", "evaluate",
           "raw_fitness", "adjusted_fitness", "ramped_half_and_half",
           "depth", "size", "to_string", "DEFAULT_FUNCTIONS"]

_MASK = 0xFFFFFFFF


class _Rng(object):
    """32-bit xorshift. Written out because R and Python do not share a
    generator and a bred population has to be reproducible in both."""

    def __init__(self, seed):
        s = int(seed) & _MASK
        self.s = s if s else 2463534242

    def next_u32(self):
        s = self.s
        s ^= (s << 13) & _MASK
        s ^= s >> 17
        s ^= (s << 5) & _MASK
        self.s = s & _MASK
        return self.s

    def unit(self):
        # 32 bits over 2^32, so the draw is in [0,1) with no rounding
        # surprise and the same bits in both arms.
        return self.next_u32() / 4294967296.0

    def below(self, n):
        """A whole number in 0..n-1, by rejection so the range is exact.
        Taking a remainder would bias the low end, and the bias differs
        with n, which is the kind of thing that never shows up in one
        language alone."""
        if n <= 1:
            return 0
        limit = _MASK - (_MASK % n)
        while True:
            v = self.next_u32()
            if v <= limit:
                return v % n


# ---------------------------------------------------------------- trees

def _fnode(op, args):
    return {"op": op, "args": args}


def _tnode(term):
    return {"term": term}


def _is_term(node):
    return "term" in node


DEFAULT_FUNCTIONS = (("+", 2), ("-", 2), ("*", 2), ("%", 2))


def _apply(op, vals):
    if op == "+":
        return vals[0] + vals[1]
    if op == "-":
        return vals[0] - vals[1]
    if op == "*":
        return vals[0] * vals[1]
    if op == "%":
        # Koza's protected division: a zero divisor yields 1, so one bad
        # division does not throw away an otherwise good program.
        return 1.0 if vals[1] == 0.0 else vals[0] / vals[1]
    raise ValueError("karpV: unknown function %r" % op)


def evaluate(node, env):
    if _is_term(node):
        t = node["term"]
        if t in env:
            return float(env[t])
        return float(t)                  # an ephemeral constant
    return _apply(node["op"], [evaluate(a, env) for a in node["args"]])


def depth(node):
    if _is_term(node):
        return 1
    return 1 + max(depth(a) for a in node["args"])


def size(node):
    if _is_term(node):
        return 1
    return 1 + sum(size(a) for a in node["args"])


def to_string(node):
    if _is_term(node):
        t = node["term"]
        if isinstance(t, float):
            # An ephemeral constant is printed through printf, not
            # through either language's own float formatting, so the two
            # arms produce the same expression text.
            return "%.17g" % t
        return str(t)
    return "(%s %s)" % (node["op"],
                        " ".join(to_string(a) for a in node["args"]))


def _random_terminal(rng, terminals, erc):
    """A terminal, or an ephemeral random constant when one is asked for.
    Koza's ERC is drawn once when the node is created and then fixed for
    the rest of the run, which is why it lives in the tree and not in a
    table."""
    n = len(terminals) + (1 if erc else 0)
    i = rng.below(n)
    if i < len(terminals):
        return _tnode(terminals[i])
    lo, hi = erc
    return _tnode(lo + (hi - lo) * rng.unit())


def _grow(rng, functions, terminals, erc, d, full):
    if d <= 1:
        return _random_terminal(rng, terminals, erc)
    if not full:
        # grow: either kind of node, chosen over the combined set, which
        # is what lets a branch stop early
        total = len(functions) + len(terminals) + (1 if erc else 0)
        if rng.below(total) >= len(functions):
            return _random_terminal(rng, terminals, erc)
    op, arity = functions[rng.below(len(functions))]
    return _fnode(op, [_grow(rng, functions, terminals, erc, d - 1, full)
                       for _ in range(arity)])


def ramped_half_and_half(rng, n, functions, terminals, erc, max_depth):
    """Depths spread evenly over 2..max_depth, half grown and half full."""
    pop = []
    span = max_depth - 1
    for i in range(n):
        d = 2 + (i % span) if span > 0 else 2
        pop.append(_grow(rng, functions, terminals, erc, d,
                         full=((i // max(span, 1)) % 2 == 1)))
    return pop


# ---------------------------------------------------------------- nodes

def _collect(node, out, path):
    out.append((path, node))
    if not _is_term(node):
        for k, a in enumerate(node["args"]):
            _collect(a, out, path + [k])
    return out


def _pick_point(rng, node, internal_bias):
    """Koza's 90/10: an internal node nine times out of ten. Uniform
    choice would pick a leaf almost every time in any sizeable tree, and
    crossover would degenerate into swapping constants."""
    nodes = _collect(node, [], [])
    internal = [p for p in nodes if not _is_term(p[1])]
    leaves = [p for p in nodes if _is_term(p[1])]
    want_internal = internal and rng.unit() < internal_bias
    pool = internal if want_internal else (leaves if leaves else internal)
    return pool[rng.below(len(pool))][0]


def _get(node, path):
    for k in path:
        node = node["args"][k]
    return node


def _replace(node, path, new):
    if not path:
        return _copy(new)
    out = _fnode(node["op"], list(node["args"]))
    out["args"][path[0]] = _replace(out["args"][path[0]], path[1:], new)
    return out


def _copy(node):
    if _is_term(node):
        return _tnode(node["term"])
    return _fnode(node["op"], [_copy(a) for a in node["args"]])


# ---------------------------------------------------------------- run

def _csum(v):
    s = 0.0
    c = 0.0
    for t in v:
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def raw_fitness(node, cases, terminals):
    """Sum of absolute errors over the fitness cases. Koza's raw fitness
    for a symbolic regression problem."""
    errs = []
    for inputs, target in cases:
        env = dict(zip(terminals, inputs))
        try:
            got = evaluate(node, env)
        except (ZeroDivisionError, OverflowError, ValueError):
            return float("inf")
        if got != got or got in (float("inf"), float("-inf")):
            return float("inf")
        errs.append(abs(got - float(target)))
    return _csum(errs)


def adjusted_fitness(raw):
    """1/(1+standardised). Compresses the difference between bad
    individuals and magnifies it between good ones, which is the whole
    reason selection uses it rather than the raw value."""
    if raw == float("inf"):
        return 0.0
    return 1.0 / (1.0 + raw)


def _roulette(rng, adj, total):
    if total <= 0.0:
        return rng.below(len(adj))
    r = rng.unit() * total
    acc = 0.0
    for i, a in enumerate(adj):
        acc += a
        if r < acc:
            return i
    return len(adj) - 1


def genetic_programming(fitness=None, ops=None, gens=20, cases=None,
                        terminals=("x",), functions=None, erc=(-5.0, 5.0),
                        pop_size=100, max_depth_init=6, max_depth=17,
                        p_crossover=0.9, p_mutation=0.0,
                        internal_bias=0.9, seed=1, elitism=0):
    """Breed a population of expression trees.

    Parameters
    ----------
    fitness : callable, optional
        Takes a tree and returns raw fitness, smaller being better. When
        absent, cases must be given and the sum of absolute errors is
        used.
    ops : sequence, optional
        Alias for functions, for the older call shape.
    gens : int
        Number of generations after the initial one.
    cases : sequence of (inputs, target)
        Fitness cases.
    terminals : sequence of str
        Variable names, matched positionally to each case's inputs.
    functions : sequence of (name, arity)
        Defaults to plus, minus, times and protected division.
    erc : (lo, hi) or None
        Range for ephemeral random constants; None to use none.
    max_depth : int
        Cap on offspring depth. A child that exceeds it is replaced by
        its first parent, which is the book's rule.
    internal_bias : float
        Probability of choosing an internal node as a crossover point.
    seed : int
        Seeds the module's own generator, so a run reproduces in both
        language arms.

    Returns
    -------
    RichResult
        best, best_string, best_raw, best_adjusted, best_size, best_depth,
        generation_found, history of (best_raw, mean_raw, best_size),
        evaluations, generations, pop_size, seed, method.
    """
    if functions is None:
        functions = ops if ops is not None else DEFAULT_FUNCTIONS
    functions = [(str(o[0]), int(o[1])) for o in functions]
    terminals = [str(t) for t in terminals]
    if fitness is None:
        if not cases:
            raise ValueError("karpV: give either a fitness function or "
                             "fitness cases")
        def fitness(tree):
            return raw_fitness(tree, cases, terminals)
    if max_depth_init < 2:
        raise ValueError("karpV: max_depth_init = %d; ramped half-and-half "
                         "needs at least 2" % max_depth_init)

    rng = _Rng(seed)
    pop = ramped_half_and_half(rng, int(pop_size), functions, terminals,
                               erc, int(max_depth_init))
    evals = 0
    best = None
    best_raw = float("inf")
    best_gen = 0
    hist = []

    for g in range(int(gens) + 1):
        raws = [fitness(t) for t in pop]
        evals += len(pop)
        adj = [adjusted_fitness(r) for r in raws]
        total = _csum(adj)
        for i, r in enumerate(raws):
            if r < best_raw:
                best_raw = r
                best = _copy(pop[i])
                best_gen = g
        finite = [r for r in raws if r != float("inf")]
        hist.append((min(raws), _csum(finite) / len(finite) if finite
                     else float("inf"),
                     size(pop[raws.index(min(raws))])))
        if g == int(gens):
            break

        nxt = []
        if elitism:
            order = sorted(range(len(pop)), key=lambda i: raws[i])
            for i in order[:int(elitism)]:
                nxt.append(_copy(pop[i]))
        while len(nxt) < len(pop):
            r = rng.unit()
            if r < p_crossover and len(pop) > 1:
                a = pop[_roulette(rng, adj, total)]
                b = pop[_roulette(rng, adj, total)]
                pa = _pick_point(rng, a, internal_bias)
                pb = _pick_point(rng, b, internal_bias)
                child = _replace(a, pa, _get(b, pb))
                if depth(child) > max_depth:
                    child = _copy(a)     # Koza: keep the parent instead
                nxt.append(child)
            elif r < p_crossover + p_mutation:
                a = pop[_roulette(rng, adj, total)]
                pa = _pick_point(rng, a, internal_bias)
                sub = _grow(rng, functions, terminals, erc,
                            int(max_depth_init), False)
                child = _replace(a, pa, sub)
                if depth(child) > max_depth:
                    child = _copy(a)
                nxt.append(child)
            else:
                nxt.append(_copy(pop[_roulette(rng, adj, total)]))
        pop = nxt[:len(pop)]

    return RichResult(payload={
        "best": best,
        "best_string": to_string(best) if best else None,
        "best_raw": best_raw,
        "best_adjusted": adjusted_fitness(best_raw),
        "best_size": size(best) if best else 0,
        "best_depth": depth(best) if best else 0,
        "generation_found": best_gen,
        "history": hist,
        "evaluations": evals,
        "generations": int(gens),
        "pop_size": int(pop_size),
        "seed": int(seed),
        "method": ("genetic programming (Koza 1992): ramped half-and-half "
                   "over depths 2..%d, fitness-proportionate selection on "
                   "adjusted fitness, %g crossover with a %g internal-node "
                   "bias, depth cap %d"
                   % (max_depth_init, p_crossover, internal_bias, max_depth)),
    })


karpV = genetic_programming


def cheatsheet():
    return ("karpV: genetic programming over expression trees (Koza "
            "1992). Ramped half-and-half init, adjusted fitness, "
            "fitness-proportionate selection, 90/10 crossover point "
            "bias, protected division. Seeded by the module's own "
            "xorshift so runs reproduce across language arms.")
