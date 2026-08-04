# morie.fn -- function file (rootcoder007/morie)
"""DPLL satisfiability search over a CNF formula.

Source CONSULTED: Davis, M., Logemann, G. & Loveland, D. (1962), "A
machine program for theorem-proving", *Communications of the ACM*
5(7):394-397.  The ACM Digital Library returned 403 to every attempt
here, so the paper itself could not be read; what is implemented is the
standard published statement of the procedure it introduced -- the
Davis-Putnam procedure with the resolution rule replaced by case
splitting:

    1. unit propagation: a one-literal clause forces its literal
    2. pure literal elimination: a variable occurring with only one
       sign may be set to satisfy every clause containing it
    3. splitting: choose an unassigned variable and recurse on both
       values; the formula is satisfiable if either branch is

Correctness is not taken on trust.  The parity harness cross-checks
every instance against an exhaustive truth-table evaluation, which is
feasible for the small formulas used there and settles satisfiability
independently of this implementation.

Branching is on the LOWEST-INDEXED unassigned variable, trying True
first.  That is a deterministic rule, so the decision and propagation
counts are reproducible and the R arm can be compared to them exactly;
a heuristic such as DLIS or VSIDS would be faster and would make the
two arms disagree on the counts while still agreeing on the answer.
"""

from ._richresult import RichResult

__all__ = ["dpll"]


def _simplify(clauses, lit):
    """Assign lit True: drop satisfied clauses, strike -lit elsewhere."""
    out = []
    for cl in clauses:
        if lit in cl:
            continue
        if -lit in cl:
            red = [x for x in cl if x != -lit]
            if not red:
                return None
            out.append(red)
        else:
            out.append(cl)
    return out


def dpll(cnf):
    """Decide satisfiability of a CNF formula by DPLL.

    Parameters
    ----------
    cnf : sequence of sequences of int
        Clauses in DIMACS literal form: a positive integer v is the
        variable v, a negative integer -v its negation.  Zero is not a
        literal.  An empty clause makes the formula unsatisfiable; an
        empty formula is satisfiable.

    Returns
    -------
    RichResult
        Keys ``satisfiable``, ``assignment`` (variable -> bool, only for
        variables the search had to fix), ``model`` (a full assignment
        over 1..n_vars, free variables set True), ``n_vars``,
        ``n_clauses``, ``decisions``, ``propagations``,
        ``pure_literals``, ``method``.
    """
    clauses = []
    for cl in cnf:
        c = [int(x) for x in cl]
        for x in c:
            if x == 0:
                raise ValueError("0 is not a literal")
        clauses.append(c)
    variables = sorted({abs(x) for cl in clauses for x in cl})
    nvars = max(variables) if variables else 0
    stats = {"decisions": 0, "propagations": 0, "pure": 0}

    def search(cls, assign):
        while True:
            if cls is None:
                return None
            if not cls:
                return assign
            unit = None
            for cl in cls:
                if len(cl) == 1:
                    unit = cl[0]
                    break
            if unit is not None:
                stats["propagations"] += 1
                assign = dict(assign)
                assign[abs(unit)] = unit > 0
                cls = _simplify(cls, unit)
                continue
            signs = {}
            for cl in cls:
                for x in cl:
                    signs.setdefault(abs(x), set()).add(x > 0)
            pure = None
            for v in sorted(signs):
                if len(signs[v]) == 1:
                    pure = v if True in signs[v] else -v
                    break
            if pure is not None:
                stats["pure"] += 1
                assign = dict(assign)
                assign[abs(pure)] = pure > 0
                cls = _simplify(cls, pure)
                continue
            break
        pick = min(abs(x) for cl in cls for x in cl)
        stats["decisions"] += 1
        for value in (True, False):
            lit = pick if value else -pick
            sub = _simplify(cls, lit)
            nxt = dict(assign)
            nxt[pick] = value
            got = search(sub, nxt)
            if got is not None:
                return got
        return None

    if any(len(cl) == 0 for cl in clauses):
        found = None
    else:
        found = search(clauses, {})
    sat = found is not None
    model = {}
    if sat:
        for v in range(1, nvars + 1):
            model[v] = found.get(v, True)
    return RichResult(
        payload={
            "satisfiable": sat,
            "assignment": found if sat else {},
            "model": model,
            "n_vars": nvars,
            "n_clauses": len(clauses),
            "decisions": stats["decisions"],
            "propagations": stats["propagations"],
            "pure_literals": stats["pure"],
            "method": "DPLL: unit propagation, pure literal, split on the "
                      "lowest-indexed variable, True first",
        }
    )


def cheatsheet():
    return "satDP: DPLL SAT solving"
