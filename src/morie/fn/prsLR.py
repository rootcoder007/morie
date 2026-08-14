# morie.fn -- function file (rootcoder007/morie)
r"""LR parsing: deciding a reduction from the left, with lookahead.

**The idea.** A bottom-up parser reads left to right and keeps a
stack of what it has seen. At each point it must decide: shift the
next token, or reduce a handle already on the stack -- and if reduce,
by which production. Knuth's result is that for an LR(k) grammar this
decision is a function of the stack contents and the next :math:`k`
tokens alone, so a finite automaton over *items* decides it.

**Items and states.** An item :math:`[A \to \alpha \cdot \beta, a]`
records a partially matched production together with a lookahead that
would justify reducing it. The closure of a set of items adds, for
every :math:`[A \to \alpha \cdot B\beta, a]` and every production
:math:`B \to \gamma`, the items :math:`[B \to \cdot\gamma, b]` for
:math:`b \in \mathrm{FIRST}(\beta a)`. States are closed item sets;
the transition on a symbol shifts the dot past it. That is the whole
construction.

**Three ways to get the lookaheads, and they differ.**

``lr1`` is Knuth's canonical construction: the lookahead is carried
per item, so a production reduced in one context can have a different
lookahead set than the same production reduced elsewhere.

``slr1`` (DeRemer 1971) throws that away and reduces
:math:`A \to \alpha` on all of :math:`\mathrm{FOLLOW}(A)`. Far fewer
states, but the FOLLOW set pools every context, so it can call for a
reduction in a context where the reduction is wrong. The grammar
:math:`S \to L = R \mid R,\; L \to *R \mid \mathrm{id},\; R \to L`
is the standard witness: SLR(1) reports a shift/reduce conflict on
``=`` that canonical LR(1) does not have, and the anchor exhibits
exactly that.

``lalr1`` merges canonical states that share a core, unioning their
lookaheads -- almost always as small as SLR and almost always as
strong as LR(1). "Almost": merging can create reduce/reduce conflicts
that neither parent had.

**Conflicts are reported, never resolved.** A shift/reduce conflict
silently resolved as shift is how the dangling ``else`` gets its
conventional meaning, and also how a parser quietly accepts a
different language than its grammar describes.

**What LR buys over LL.** Left recursion is fine here -- the handle is
recognised after it is complete -- so :math:`E \to E - T` can be
written directly and the parse tree comes out left-associative. The
anchor evaluates ``id - id - id`` and gets :math:`(1-2)-3`, not
:math:`1-(2-3)`.

Grammars are those of :mod:`morie.fn.prsLL`.

References
----------
Knuth, D. E. (1965) "On the translation of languages from left to
right", *Information and Control* 8(6), 607-639,
doi:10.1016/S0019-9958(65)90426-2. The LR(k) condition, the item
sets and the finite-state characterisation of the shift/reduce
decision reproduced above.

DeRemer, F. L. (1971) "Simple LR(k) grammars", *Communications of the
ACM* 14(7), 453-460, doi:10.1145/362619.362625, for the SLR
construction that takes reduction lookaheads from FOLLOW, and for the
core-merging idea behind LALR.
"""

from ._richresult import RichResult
from .prsLL import (END, EPSILON, first_sets, follow_sets, grammar,
                    linearise, nonterminals, terminals)

__all__ = ["augment", "closure", "goto", "canonical_collection",
           "build_tables", "conflicts", "parse", "lr_parser",
           "METHODS"]

METHODS = ("lr1", "slr1", "lalr1")
AUG = "S'"


def augment(g):
    r"""Add :math:`S' \to S` so acceptance is a single reduction."""
    tag = AUG
    while tag in nonterminals(g):
        tag += "'"
    return {"rules": [(tag, (g["start"],))] + list(g["rules"]),
            "start": tag, "original_start": g["start"]}


def _first_seq(seq, first, nts):
    out = set()
    for s in seq:
        if s not in nts:
            out.add(s)
            return out
        out |= first[s] - {EPSILON}
        if EPSILON not in first[s]:
            return out
    out.add(EPSILON)
    return out


def closure(items, ag, first, nts, k=1):
    r"""Close an item set. ``k=0`` gives LR(0) items (no lookahead)."""
    out = set(items)
    changed = True
    while changed:
        changed = False
        for it in list(out):
            i, dot = it[0], it[1]
            _, rhs = ag["rules"][i]
            if dot >= len(rhs) or rhs[dot] not in nts:
                continue
            B = rhs[dot]
            if k == 0:
                looks = (None,)
            else:
                tail = rhs[dot + 1:]
                fs = _first_seq(tail, first, nts)
                looks = (fs - {EPSILON}) | ({it[2]} if EPSILON in fs
                                            or not tail else set())
            for j, (lhs, _) in enumerate(ag["rules"]):
                if lhs != B:
                    continue
                for b in looks:
                    new = (j, 0) if k == 0 else (j, 0, b)
                    if new not in out:
                        out.add(new)
                        changed = True
    return frozenset(out)


def goto(state, sym, ag, first, nts, k=1):
    r"""The state reached by shifting the dot past ``sym``."""
    moved = set()
    for it in state:
        i, dot = it[0], it[1]
        _, rhs = ag["rules"][i]
        if dot < len(rhs) and rhs[dot] == sym:
            moved.add((i, dot + 1) if k == 0 else (i, dot + 1, it[2]))
    return closure(moved, ag, first, nts, k) if moved else frozenset()


def canonical_collection(ag, k=1):
    r"""Every reachable state, and the transitions between them."""
    g0 = {"rules": ag["rules"], "start": ag["start"]}
    first = first_sets(g0)
    nts = set(nonterminals(g0))
    syms = list(nts) + terminals(g0)
    start_item = (0, 0) if k == 0 else (0, 0, END)
    I0 = closure({start_item}, ag, first, nts, k)
    states, index = [I0], {I0: 0}
    trans = {}
    q = [I0]
    while q:
        I = q.pop(0)
        for X in syms:
            J = goto(I, X, ag, first, nts, k)
            if not J:
                continue
            if J not in index:
                index[J] = len(states)
                states.append(J)
                q.append(J)
            trans[(index[I], X)] = index[J]
    return {"states": states, "index": index, "transitions": trans,
            "first": first, "nonterminals": nts}


def _core(state):
    return frozenset((i, d) for i, d, _ in state)


def build_tables(g, method="lr1"):
    r"""ACTION and GOTO, with every conflict recorded."""
    if method not in METHODS:
        raise ValueError("prsLR: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    ag = augment(g)
    k = 0 if method == "slr1" else 1
    col = canonical_collection(ag, k)
    states, trans = col["states"], col["transitions"]
    nts = col["nonterminals"]
    follow = follow_sets({"rules": ag["rules"], "start": ag["start"]},
                         col["first"]) if method == "slr1" else None

    if method == "lalr1":
        groups = {}
        for n, st in enumerate(states):
            groups.setdefault(_core(st), []).append(n)
        remap, merged = {}, []
        for core, members in groups.items():
            new = len(merged)
            union = set()
            for n in members:
                remap[n] = new
                union |= set(states[n])
            merged.append(frozenset(union))
        states = merged
        trans = {(remap[s], X): remap[t]
                 for (s, X), t in trans.items()}

    action, gotos, confl = {}, {}, []

    def put(s, a, act):
        if (s, a) in action and action[(s, a)] != act:
            confl.append({"state": s, "lookahead": a,
                          "existing": action[(s, a)], "proposed": act,
                          "kind": ("shift/reduce"
                                   if "shift" in (action[(s, a)][0],
                                                  act[0])
                                   else "reduce/reduce")})
        else:
            action[(s, a)] = act

    for (s, X), t in trans.items():
        if X in nts:
            gotos[(s, X)] = t
        else:
            put(s, X, ("shift", t))
    for s, st in enumerate(states):
        for it in st:
            i, dot = it[0], it[1]
            lhs, rhs = ag["rules"][i]
            if dot != len(rhs):
                continue
            if i == 0:
                put(s, END, ("accept", None))
                continue
            if method == "slr1":
                looks = follow[lhs]
            else:
                looks = {it[2]}
            for a in looks:
                put(s, a, ("reduce", i))
    return {"action": action, "goto": gotos, "states": states,
            "n_states": len(states), "conflicts": confl,
            "rules": ag["rules"], "augmented": ag, "method": method}


def conflicts(g, method="lr1"):
    r"""The conflicts a given construction produces on a grammar."""
    t = build_tables(g, method)
    return RichResult(payload={
        "estimate": t["conflicts"], "conflicts": t["conflicts"],
        "n_conflicts": len(t["conflicts"]), "method": method,
        "n_states": t["n_states"],
        "ok": not t["conflicts"],
    })


def _leaf(sym):
    return {"symbol": sym, "children": None}


def parse(g, tokens, method="lr1", tables=None):
    r"""Shift-reduce parse, returning the parse tree."""
    t = tables if tables is not None else build_tables(g, method)
    if t["conflicts"]:
        c = t["conflicts"][0]
        raise ValueError("prsLR: the grammar is not %s -- %d "
                         "conflict(s), first a %s in state %d on %r"
                         % (t["method"], len(t["conflicts"]),
                            c["kind"], c["state"], c["lookahead"]))
    toks = [str(x) for x in tokens] + [END]
    stack, trees, pos = [0], [], 0
    for _ in range(100000):
        a = toks[pos]
        act = t["action"].get((stack[-1], a))
        if act is None:
            raise ValueError("prsLR: syntax error at token %d (%r) "
                             "in state %d" % (pos, a, stack[-1]))
        if act[0] == "shift":
            stack.append(act[1])
            trees.append(_leaf(a))
            pos += 1
        elif act[0] == "reduce":
            lhs, rhs = t["rules"][act[1]]
            kids = []
            for _s in rhs:
                stack.pop()
                kids.append(trees.pop())
            kids.reverse()
            node = {"symbol": lhs, "children": kids}
            nxt = t["goto"].get((stack[-1], lhs))
            if nxt is None:
                raise ValueError("prsLR: no goto for %r in state %d"
                                 % (lhs, stack[-1]))
            stack.append(nxt)
            trees.append(node)
        else:
            if len(trees) != 1 or pos != len(toks) - 1:
                raise ValueError("prsLR: accepted with %d trees and "
                                 "%d tokens left"
                                 % (len(trees), len(toks) - 1 - pos))
            return trees[0]
    raise ValueError("prsLR: the parser did not terminate")


def lr_parser(grammar_, tokens, method="lr1"):
    r"""Entry point: parse ``tokens`` bottom-up."""
    g = grammar_ if isinstance(grammar_, dict) else grammar(grammar_)
    t = build_tables(g, method)
    tree = parse(g, tokens, method, t)
    return RichResult(payload={
        "estimate": tree, "tree": tree, "method": method,
        "n_states": t["n_states"], "conflicts": t["conflicts"],
        "tokens": [str(x) for x in tokens],
        "yield": linearise(tree),
    })
