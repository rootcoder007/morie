# morie.fn -- function file (rootcoder007/morie)
r"""Term rewriting: normal forms, confluence, and completion.

**Rewriting.** A rule :math:`l \to r` applies to a term when some
subterm *matches* :math:`l` -- matching, not unification: the rule's
variables may be bound, the term's may not. Replace that subterm by
:math:`r` under the same binding and repeat. A term with no
applicable rule is in **normal form**.

**Two things can go wrong, and they are independent.**

*Termination.* :math:`f(x) \to f(f(x))` never stops. Termination is
undecidable in general, so a *reduction order* is used instead: if
every rule strictly decreases its term in a well-founded order,
rewriting must halt. The lexicographic path order (LPO) is provided
and is what orients rules during completion.

*Confluence.* :math:`a \to b` and :math:`a \to c` both apply to
:math:`a`, and the results never meet again -- so the normal form
depends on which rule was picked, and the system decides nothing.

**Where non-confluence hides.** Two rules can only interfere where
their left-hand sides overlap. Superimpose :math:`l_2` on a
non-variable subterm of :math:`l_1` by unification; the two ways of
rewriting the overlap give a **critical pair**. Knuth and Bendix's
Critical Pair Lemma is that a system is *locally* confluent exactly
when every critical pair is joinable -- finitely many checks instead
of infinitely many terms. Newman's lemma then upgrades local
confluence to confluence for terminating systems, which is why
termination is checked and not assumed.

**Completion.** When a critical pair is not joinable, orient it into
a new rule with the reduction order and add it; its own critical
pairs join the queue. On success the result is *convergent* --
terminating and confluent -- so equality of two terms is decided by
reducing both to normal form. The procedure may also loop forever,
or stop unable to orient a pair (commutativity :math:`x + y = y + x`
cannot be oriented by any reduction order); both outcomes are
reported rather than hidden.

Terms are those of :mod:`morie.fn.unifAlg`.

References
----------
Knuth, D. E. & Bendix, P. B. (1970) "Simple word problems in
universal algebras", in J. Leech (ed.) *Computational Problems in
Abstract Algebra*, Pergamon Press, 263-297,
doi:10.1016/B978-0-08-012975-4.50028-X. The superposition of one
left-hand side on a non-variable subterm of another, the Critical
Pair Lemma, the completion procedure reproduced above, and the
failure case of an unorientable pair.

Newman, M. H. A. (1942) "On theories with a combinatorial definition
of 'equivalence'", *Annals of Mathematics* 43(2), 223-243,
doi:10.2307/1968867, for the lemma that a terminating, locally
confluent relation is confluent.

Baader, F. & Nipkow, T. (1998) *Term Rewriting and All That*,
Cambridge University Press, ISBN 978-0-521-77920-3. Ch. 2 (abstract
reduction systems, normal forms), Ch. 5 (the lexicographic path
order and its subterm/precedence cases as implemented in
:func:`lpo_greater`), and Ch. 7 (critical pairs and completion).
"""

from ._richresult import RichResult
from .unifAlg import (VAR, app, apply_subst, const, is_var, match,
                      substitute, unify, var, variables)

__all__ = ["rule", "positions", "subterm_at", "replace_at",
           "rewrite_step", "normal_form", "lpo_greater",
           "critical_pairs", "joinable", "is_locally_confluent",
           "is_terminating", "complete", "decides",
           "term_rewriting"]

STRATEGIES = ("innermost", "outermost")


def rule(lhs, rhs):
    r"""A rewrite rule, checked for the two conditions rules need."""
    if is_var(lhs):
        raise ValueError("trmRew: a rule cannot have a bare variable "
                         "on the left -- it would match everything")
    extra = set(variables(rhs)) - set(variables(lhs))
    if extra:
        raise ValueError("trmRew: the right-hand side introduces the "
                         "unbound variable(s) %s"
                         % ", ".join(sorted(extra)))
    return (lhs, rhs)


def positions(t):
    r"""Every position in a term, as a tuple of argument indices."""
    out = [()]
    if not is_var(t):
        for i, a in enumerate(t[2]):
            out.extend((i,) + p for p in positions(a))
    return out


def subterm_at(t, pos):
    r"""The subterm at a position."""
    cur = t
    for i in pos:
        if is_var(cur) or i >= len(cur[2]):
            raise ValueError("trmRew: position %r does not exist in "
                             "the term" % (pos,))
        cur = cur[2][i]
    return cur


def replace_at(t, pos, new):
    r"""The term with the subterm at ``pos`` replaced."""
    if not pos:
        return new
    if is_var(t) or pos[0] >= len(t[2]):
        raise ValueError("trmRew: position %r does not exist in the "
                         "term" % (pos,))
    args = list(t[2])
    args[pos[0]] = replace_at(args[pos[0]], pos[1:], new)
    return app(t[1], *args)


def rewrite_step(t, rules, strategy="innermost"):
    r"""One rewrite, or ``None`` when the term is in normal form.

    Innermost reduces arguments before the term above them; outermost
    the other way round. For a convergent system both reach the same
    normal form, which the anchor checks.
    """
    if strategy not in STRATEGIES:
        raise ValueError("trmRew: strategy must be one of %s, got %r"
                         % (", ".join(STRATEGIES), strategy))
    pos = positions(t)
    pos.sort(key=len, reverse=(strategy == "innermost"))
    for p in pos:
        s = subterm_at(t, p)
        if is_var(s):
            continue
        for i, (l, r) in enumerate(rules):
            m = match(l, s)
            if m is not None:
                return {"term": replace_at(t, p, substitute(r, m)),
                        "position": p, "rule": i, "binding": m}
    return None


def normal_form(t, rules, strategy="innermost", max_steps=10000):
    r"""Rewrite to exhaustion. Raises if the step budget runs out --
    silently returning a half-reduced term would hide a loop.
    """
    cur, trace = t, []
    for _ in range(int(max_steps)):
        st = rewrite_step(cur, rules, strategy)
        if st is None:
            return {"normal_form": cur, "steps": len(trace),
                    "trace": trace}
        trace.append((st["rule"], st["position"]))
        cur = st["term"]
    raise ValueError("trmRew: no normal form after %d steps -- the "
                     "system does not terminate on this term"
                     % int(max_steps))


def _prec(precedence, sym):
    return precedence.get(sym, 0)


def lpo_greater(s, t, precedence):
    r"""The lexicographic path order, :math:`s >_{lpo} t`.

    A well-founded order on terms: a rule that strictly decreases in
    it cannot be applied forever.
    """
    if s == t:
        return False
    if is_var(t):
        return not is_var(s) and t[1] in variables(s)
    if is_var(s):
        return False
    if any(a == t or lpo_greater(a, t, precedence) for a in s[2]):
        return True
    ps, pt = _prec(precedence, s[1]), _prec(precedence, t[1])
    if ps > pt:
        return all(lpo_greater(s, b, precedence) for b in t[2])
    if ps < pt:
        return False
    if len(s[2]) != len(t[2]):
        return len(s[2]) > len(t[2])
    for a, b in zip(s[2], t[2]):
        if a == b:
            continue
        return (lpo_greater(a, b, precedence)
                and all(lpo_greater(s, c, precedence) for c in t[2]))
    return False


def is_terminating(rules, precedence):
    r"""Whether every rule strictly decreases in the LPO.

    Sufficient, not necessary: a ``False`` means this order does not
    prove termination, not that the system loops.
    """
    bad = [i for i, (l, r) in enumerate(rules)
           if not lpo_greater(l, r, precedence)]
    return {"terminating": not bad, "unoriented": bad,
            "method": "lexicographic path order (Baader & Nipkow "
                      "1998 Ch. 5); sufficient, not necessary"}


def _rename(t, tag):
    if is_var(t):
        return (VAR, t[1] + tag)
    return app(t[1], *[_rename(a, tag) for a in t[2]])


def _overlap(ra, rb, same):
    r"""Superpose ``rb``'s left-hand side on ``ra``'s, at every
    non-variable position, and rewrite the overlap both ways.
    """
    l1, r1 = ra
    L2, R2 = _rename(rb[0], "#2"), _rename(rb[1], "#2")
    out = []
    for p in positions(l1):
        if same and p == ():
            continue  # a rule overlaps itself trivially at the root
        sub_t = subterm_at(l1, p)
        if is_var(sub_t):
            continue
        u = unify(sub_t, L2)
        if not u["unified"]:
            continue
        sig = u["mgu"]
        a = apply_subst(r1, sig)
        b = apply_subst(replace_at(l1, p, R2), sig)
        if a != b:
            out.append({"left": a, "right": b, "position": p})
    return out


def critical_pairs(rules):
    r"""Every overlap between two left-hand sides.

    Knuth & Bendix's superposition: unify a non-variable subterm of
    one left-hand side with another, then rewrite the overlap both
    ways.
    """
    out = []
    for i, ra in enumerate(rules):
        for j, rb in enumerate(rules):
            for c in _overlap(ra, rb, i == j):
                c["rules"] = (i, j)
                out.append(c)
    return out


def joinable(a, b, rules, max_steps=10000):
    r"""Whether two terms reach a common normal form."""
    try:
        na = normal_form(a, rules, max_steps=max_steps)["normal_form"]
        nb = normal_form(b, rules, max_steps=max_steps)["normal_form"]
    except ValueError:
        return False
    return na == nb


def is_locally_confluent(rules, max_steps=10000):
    r"""The Critical Pair Lemma, applied."""
    cps = critical_pairs(rules)
    bad = [c for c in cps
           if not joinable(c["left"], c["right"], rules, max_steps)]
    return RichResult(payload={
        "estimate": not bad, "locally_confluent": not bad,
        "n_critical_pairs": len(cps), "unjoinable": bad,
        "method": "Knuth & Bendix (1970) Critical Pair Lemma",
    })


def is_confluent(rules, precedence, max_steps=10000):
    r"""Confluence via Newman's lemma: terminating and locally
    confluent.
    """
    term = is_terminating(rules, precedence)
    lc = is_locally_confluent(rules, max_steps)
    yes = term["terminating"] and lc["locally_confluent"]
    return RichResult(payload={
        "estimate": yes, "confluent": yes,
        "terminating": term["terminating"],
        "locally_confluent": lc["locally_confluent"],
        "n_critical_pairs": lc["n_critical_pairs"],
        "unjoinable": lc["unjoinable"],
        "method": "Newman (1942): terminating + locally confluent "
                  "implies confluent",
    })


def complete(equations, precedence, max_rules=60, max_steps=10000,
             max_iter=4000):
    r"""Knuth-Bendix completion of a set of equations.

    Huet's form of the procedure: rules are kept interreduced as they
    are found, and only the critical pairs a *new* rule creates are
    queued. Recomputing every pair from scratch each round makes the
    group axioms alone take minutes.

    Reports success, an unorientable pair, or exhaustion of the rule
    budget -- the procedure is a semi-decision procedure and saying
    otherwise would be a lie.
    """
    rules = []
    queue = [(l, r) for l, r in equations]
    for _ in range(int(max_iter)):
        if not queue:
            rules = _interreduce(rules, precedence)
            return RichResult(payload={
                "estimate": rules, "rules": rules, "complete": True,
                "reason": None, "n_rules": len(rules),
                "method": "Knuth & Bendix (1970) completion, oriented "
                          "by the lexicographic path order",
            })
        s_, t_ = queue.pop(0)
        s_ = normal_form(s_, rules, max_steps=max_steps)["normal_form"]
        t_ = normal_form(t_, rules, max_steps=max_steps)["normal_form"]
        if s_ == t_:
            continue
        if lpo_greater(s_, t_, precedence):
            new = rule(s_, t_)
        elif lpo_greater(t_, s_, precedence):
            new = rule(t_, s_)
        else:
            return _incomplete(rules, "unorientable equation"
                               if not rules else
                               "unorientable critical pair", (s_, t_))
        # Collapse: a rule whose left-hand side the new rule can
        # rewrite is no longer a rule; it goes back into the queue.
        keep = []
        for l, r in rules:
            if rewrite_step(l, [new]) is not None:
                queue.append((l, r))
            else:
                keep.append((l, normal_form(
                    r, [new], max_steps=max_steps)["normal_form"]))
        rules = keep + [new]
        if len(rules) > int(max_rules):
            return _incomplete(rules, "rule budget exhausted", None)
        for other in rules:
            for c in _overlap(new, other, new is other):
                queue.append((c["left"], c["right"]))
            if other is not new:
                for c in _overlap(other, new, False):
                    queue.append((c["left"], c["right"]))
    return _incomplete(rules, "iteration budget exhausted", None)


def _incomplete(rules, why, pair):
    return RichResult(payload={
        "estimate": None, "rules": rules, "complete": False,
        "reason": why, "pair": pair, "n_rules": len(rules),
        "method": "Knuth & Bendix (1970) completion, oriented by the "
                  "lexicographic path order",
    })


def _canonical(rules):
    """Rename variables to x0, x1, ... so completion output does not
    carry the bookkeeping suffixes renaming apart introduced."""
    out = []
    for l, r in rules:
        names = variables(l)
        sub = {n: var("x%d" % i) for i, n in enumerate(names)}
        out.append((substitute(l, sub), substitute(r, sub)))
    return out


def _interreduce(rules, precedence):
    out = list(rules)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            rest = out[:i] + out[i + 1:]
            l, r = out[i]
            nr = (normal_form(r, rest)["normal_form"] if rest else r)
            if rest and rewrite_step(l, rest) is not None:
                out = rest
                changed = True
                break
            if nr != r:
                out[i] = (l, nr)
                changed = True
                break
    return _canonical(out)


def decides(s, t, rules, max_steps=10000):
    r"""Whether two terms are equal in the theory the rules present.

    Sound only for a convergent system; that is what completion is
    for.
    """
    a = normal_form(s, rules, max_steps=max_steps)["normal_form"]
    b = normal_form(t, rules, max_steps=max_steps)["normal_form"]
    return {"equal": a == b, "left": a, "right": b}


def term_rewriting(term, rules, strategy="innermost",
                   max_steps=10000):
    r"""Entry point: reduce ``term`` under ``rules``."""
    nf = normal_form(term, rules, strategy, max_steps)
    return RichResult(payload={
        "estimate": nf["normal_form"], "normal_form": nf["normal_form"],
        "steps": nf["steps"], "trace": nf["trace"],
        "strategy": strategy,
        "method": "leftmost-%s rewriting to normal form" % strategy,
    })


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
termrewriting = term_rewriting
