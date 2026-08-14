# morie.fn -- function file (rootcoder007/morie)
r"""LL(1) parsing: the table, and what top-down analysis cannot do.

**The decision.** A top-down parser expands the leftmost nonterminal.
Sitting at :math:`A` with one lookahead token :math:`a`, it must pick
a production :math:`A \to \alpha` *before* seeing what :math:`\alpha`
derives. That choice is possible for every :math:`(A, a)` exactly
when the grammar is LL(1), and the two ingredients are

.. math:: \mathrm{FIRST}(\alpha) = \{a : \alpha
          \Rightarrow^{*} a\beta\}, \qquad
          \mathrm{FOLLOW}(A) = \{a : S \Rightarrow^{*}
          \alpha A a \beta\}.

Production :math:`A \to \alpha` is entered under every
:math:`a \in \mathrm{FIRST}(\alpha)`, and additionally under every
:math:`a \in \mathrm{FOLLOW}(A)` when :math:`\alpha` can derive the
empty string. Two productions landing in one cell is a conflict, and
the grammar is not LL(1) -- this is reported with the offending pair
rather than silently resolved, because silently resolving it is how a
parser comes to accept a language nobody wrote down.

**The structural obstruction.** Left recursion :math:`A \to A\alpha`
puts :math:`\mathrm{FIRST}(A\alpha) \subseteq
\mathrm{FIRST}(A)`, so the recursive production and the base
production compete on the same lookahead, always. No amount of
lookahead fixes it: a top-down parser expanding :math:`A` would
expand :math:`A` again with no input consumed. This is the price
top-down analysis pays, and the standard transformation to right
recursion is provided -- it changes the parse tree's shape, so the
associativity a left-recursive grammar encoded has to be recovered
some other way.

**Two routes, one answer.** The table-driven parser runs an explicit
stack; the recursive-descent parser uses the call stack and reads
like the grammar. Both are implemented and must produce identical
trees, which the anchor checks -- one of them being wrong is much
easier than both being wrong the same way.

References
----------
Knuth, D. E. (1971) "Top-down syntax analysis", *Acta Informatica*
1(2), 79-110, doi:10.1007/BF00289517. Top-down (LL) analysis, the
role of one-symbol lookahead, and the failure of top-down methods on
left-recursive grammars.

Knuth, D. E. (1965) "On the translation of languages from left to
right", *Information and Control* 8(6), 607-639,
doi:10.1016/S0019-9958(65)90426-2, for the LR(k) classes against
which the LL(1) restriction is measured; see
:mod:`morie.fn.prsLR`.
"""

from ._richresult import RichResult

__all__ = ["grammar", "nonterminals", "terminals", "first_sets",
           "first_of", "follow_sets", "ll1_table", "is_ll1",
           "left_recursive", "remove_left_recursion", "parse",
           "linearise", "ll_parser"]

EPSILON = ""
END = "$"
ROUTES = ("table", "recursive_descent")


def grammar(rules, start=None):
    r"""A context-free grammar as a list of ``(lhs, [symbols])``.

    An empty right-hand side is the empty production. Any symbol that
    never appears on a left-hand side is a terminal.
    """
    R = []
    for item in rules:
        lhs, rhs = item
        if not isinstance(lhs, str) or not lhs:
            raise ValueError("prsLL: a left-hand side must be a "
                             "non-empty symbol, got %r" % (lhs,))
        seq = tuple(str(s) for s in rhs)
        if any(s == "" for s in seq):
            raise ValueError("prsLL: write the empty production as an "
                             "empty right-hand side, not as %r" % ("",))
        if END in seq or lhs == END:
            raise ValueError("prsLL: %r is reserved for end of input"
                             % END)
        R.append((lhs, seq))
    if not R:
        raise ValueError("prsLL: the grammar has no productions")
    S = R[0][0] if start is None else str(start)
    if S not in {l for l, _ in R}:
        raise ValueError("prsLL: the start symbol %r has no "
                         "production" % S)
    g = {"rules": R, "start": S}
    unreachable = set(nonterminals(g)) - _reachable(g)
    if unreachable:
        raise ValueError("prsLL: nonterminal(s) %s cannot be reached "
                         "from the start symbol"
                         % ", ".join(sorted(unreachable)))
    return g


def _reachable(g):
    seen, stack = {g["start"]}, [g["start"]]
    nts = set(nonterminals(g))
    while stack:
        A = stack.pop()
        for lhs, rhs in g["rules"]:
            if lhs != A:
                continue
            for s in rhs:
                if s in nts and s not in seen:
                    seen.add(s)
                    stack.append(s)
    return seen


def nonterminals(g):
    r"""Every symbol with a production, in order of first appearance."""
    out = []
    for lhs, _ in g["rules"]:
        if lhs not in out:
            out.append(lhs)
    return out


def terminals(g):
    r"""Every symbol without one."""
    nts = set(nonterminals(g))
    out = []
    for _, rhs in g["rules"]:
        for s in rhs:
            if s not in nts and s not in out:
                out.append(s)
    return out


def first_sets(g):
    r"""FIRST for every nonterminal; ``EPSILON`` marks nullability."""
    nts = set(nonterminals(g))
    first = {A: set() for A in nts}
    changed = True
    while changed:
        changed = False
        for A, rhs in g["rules"]:
            add = _first_seq(rhs, first, nts)
            if not add <= first[A]:
                first[A] |= add
                changed = True
    return first


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


def first_of(seq, g, first=None):
    r"""FIRST of an arbitrary string of symbols."""
    f = first_sets(g) if first is None else first
    return _first_seq(tuple(seq), f, set(nonterminals(g)))


def follow_sets(g, first=None):
    r"""FOLLOW for every nonterminal, with ``END`` in FOLLOW(start)."""
    nts = set(nonterminals(g))
    f = first_sets(g) if first is None else first
    follow = {A: set() for A in nts}
    follow[g["start"]].add(END)
    changed = True
    while changed:
        changed = False
        for A, rhs in g["rules"]:
            for i, s in enumerate(rhs):
                if s not in nts:
                    continue
                rest = _first_seq(rhs[i + 1:], f, nts)
                add = rest - {EPSILON}
                if EPSILON in rest or i + 1 == len(rhs):
                    add |= follow[A]
                if not add <= follow[s]:
                    follow[s] |= add
                    changed = True
    return follow


def ll1_table(g):
    r"""The predictive table, plus every conflict found building it."""
    first = first_sets(g)
    follow = follow_sets(g, first)
    nts = set(nonterminals(g))
    table, conflicts = {}, []
    for i, (A, rhs) in enumerate(g["rules"]):
        look = _first_seq(rhs, first, nts)
        cells = set(look - {EPSILON})
        if EPSILON in look:
            cells |= follow[A]
        for a in cells:
            if (A, a) in table and table[(A, a)] != i:
                conflicts.append({"nonterminal": A, "lookahead": a,
                                  "rules": (table[(A, a)], i)})
            else:
                table[(A, a)] = i
    return {"table": table, "conflicts": conflicts, "first": first,
            "follow": follow}


def is_ll1(g):
    r"""Whether one token of lookahead suffices everywhere."""
    t = ll1_table(g)
    return RichResult(payload={
        "estimate": not t["conflicts"], "ll1": not t["conflicts"],
        "conflicts": t["conflicts"], "table": t["table"],
        "first": t["first"], "follow": t["follow"],
        "left_recursive": left_recursive(g),
        "method": "Knuth (1971): FIRST/FOLLOW table, one production "
                  "per (nonterminal, lookahead) cell",
    })


def left_recursive(g):
    r"""Nonterminals that derive themselves leftmost, directly or not.

    Indirect recursion counts: a top-down parser loops on it just the
    same.
    """
    nts = set(nonterminals(g))
    first = first_sets(g)
    edges = {A: set() for A in nts}
    for A, rhs in g["rules"]:
        for s in rhs:
            if s not in nts:
                break
            edges[A].add(s)
            if EPSILON not in first[s]:
                break
    out = []
    for A in nonterminals(g):
        seen, stack = set(), [A]
        while stack:
            B = stack.pop()
            for C in edges[B]:
                if C == A:
                    out.append(A)
                    stack = []
                    break
                if C not in seen:
                    seen.add(C)
                    stack.append(C)
            else:
                continue
            break
    return out


def remove_left_recursion(g):
    r"""Rewrite direct left recursion into right recursion.

    The language is preserved; the parse trees are not. A grammar
    that encoded left associativity through its recursion loses it
    here, which is why this is offered rather than applied.
    """
    rules = []
    nts = nonterminals(g)
    for A in nts:
        prods = [rhs for lhs, rhs in g["rules"] if lhs == A]
        rec = [p[1:] for p in prods if p and p[0] == A]
        base = [p for p in prods if not (p and p[0] == A)]
        if not rec:
            rules.extend((A, p) for p in prods)
            continue
        if not base:
            raise ValueError("prsLL: %r is left-recursive with no "
                             "base production, so it derives nothing"
                             % A)
        tail = A + "'"
        while tail in nts:
            tail += "'"
        rules.extend((A, p + (tail,)) for p in base)
        rules.extend((tail, p + (tail,)) for p in rec)
        rules.append((tail, ()))
    return grammar(rules, g["start"])


def _leaf(sym):
    return {"symbol": sym, "children": None}


def _node(sym, kids):
    return {"symbol": sym, "children": kids}


def parse(g, tokens, route="table"):
    r"""Parse a token list, returning the parse tree.

    ``route`` selects the table-driven stack machine or straight
    recursive descent; both must agree.
    """
    if route not in ROUTES:
        raise ValueError("prsLL: route must be one of %s, got %r"
                         % (", ".join(ROUTES), route))
    t = ll1_table(g)
    if t["conflicts"]:
        raise ValueError("prsLL: the grammar is not LL(1) -- %d "
                         "conflict(s), first at (%s, %r)"
                         % (len(t["conflicts"]),
                            t["conflicts"][0]["nonterminal"],
                            t["conflicts"][0]["lookahead"]))
    toks = [str(x) for x in tokens] + [END]
    if route == "table":
        tree, pos = _parse_table(g, t["table"], toks)
    else:
        tree, pos = _parse_rd(g, t["table"], toks, g["start"], 0)
    if pos != len(toks) - 1:
        raise ValueError("prsLL: input not consumed -- stopped at "
                         "token %d (%r)" % (pos, toks[pos]))
    return tree


def _pick(table, A, a):
    if (A, a) not in table:
        raise ValueError("prsLL: no production for %r on lookahead "
                         "%r" % (A, a))
    return table[(A, a)]


def _parse_rd(g, table, toks, A, pos):
    _, rhs = g["rules"][_pick(table, A, toks[pos])]
    nts = set(nonterminals(g))
    kids = []
    for s in rhs:
        if s in nts:
            sub, pos = _parse_rd(g, table, toks, s, pos)
            kids.append(sub)
        else:
            if toks[pos] != s:
                raise ValueError("prsLL: expected %r but found %r at "
                                 "token %d" % (s, toks[pos], pos))
            kids.append(_leaf(s))
            pos += 1
    return _node(A, kids), pos


def _parse_table(g, table, toks):
    nts = set(nonterminals(g))
    root = _node(g["start"], [])
    stack = [(g["start"], root)]
    pos = 0
    while stack:
        sym, node = stack.pop()
        if sym in nts:
            _, rhs = g["rules"][_pick(table, sym, toks[pos])]
            kids = [(_node(s, []) if s in nts else _leaf(s))
                    for s in rhs]
            node["children"] = kids
            for s, k in reversed(list(zip(rhs, kids))):
                stack.append((s, k))
        else:
            if toks[pos] != sym:
                raise ValueError("prsLL: expected %r but found %r at "
                                 "token %d" % (sym, toks[pos], pos))
            pos += 1
    return root, pos


def linearise(tree):
    r"""The terminals of a parse tree, left to right."""
    if tree["children"] is None:
        return [tree["symbol"]]
    out = []
    for k in tree["children"]:
        out.extend(linearise(k))
    return out


def ll_parser(grammar_, tokens, route="table"):
    r"""Entry point: parse ``tokens`` under an LL(1) grammar."""
    g = grammar_ if isinstance(grammar_, dict) else grammar(grammar_)
    tree = parse(g, tokens, route)
    return RichResult(payload={
        "estimate": tree, "tree": tree, "route": route,
        "tokens": [str(x) for x in tokens],
        "yield": linearise(tree),
        "method": "Knuth (1971) top-down analysis with one token of "
                  "lookahead",
    })
