# morie.fn -- function file (rootcoder007/morie)
r"""Robinson unification, and the check that is usually left out.

**The problem.** Two terms unify when some substitution makes them
identical. Robinson's contribution was not that such a substitution
can be searched for, but that when one exists there is a *most
general* one, and that it is computable: every other unifier factors
through it.

**The disagreement set.** Robinson's algorithm walks the two terms in
parallel and, at the first position where they differ, extracts the
pair of subterms sitting there -- the disagreement set. If neither is
a variable the terms cannot be unified. If one is a variable
:math:`v` and the other a term :math:`t`, bind :math:`v \mapsto t`,
apply the binding everywhere, and repeat. The algorithm terminates
because each round removes one variable from the problem.

**The occurs check.** Binding :math:`v \mapsto t` is legitimate only
when :math:`v` does not occur inside :math:`t`. Unifying :math:`x`
with :math:`f(x)` would otherwise produce the "solution"
:math:`x \mapsto f(x)`, whose repeated application never reaches a
finite term -- :math:`f(f(f(\ldots)))`. Robinson's Sec. 5 has the
check; most Prolog systems omit it for speed and are unsound as a
result, so ``occurs_check=False`` is offered and documented rather
than hidden.

**Most general, and what that buys.** If :math:`\sigma` is the
returned unifier and :math:`\theta` any other, there is a
:math:`\delta` with :math:`\theta = \delta \circ \sigma`. So nothing
is decided prematurely: resolution can commit to :math:`\sigma` and
still reach every conclusion reachable through any other unifier.
``factor_through`` computes that :math:`\delta` and the anchor uses
it, rather than asserting generality on faith.

**Terms.** A variable is ``("VAR", name)``; an application is
``("APP", symbol, (arg, ...))``. Constants are applications of
arity zero. Build them with :func:`var`, :func:`app` and
:func:`const` rather than by hand.

References
----------
Robinson, J. A. (1965) "A Machine-Oriented Logic Based on the
Resolution Principle", *Journal of the ACM* 12(1), 23-41,
doi:10.1145/321250.321253. Sec. 5 (the Unification Theorem: any
unifiable set has a most general unifier), the disagreement-set
algorithm, the occurs check, and the factorisation
:math:`\theta = \delta \circ \sigma` reproduced above.
"""

from ._richresult import RichResult

__all__ = ["var", "app", "const", "is_var", "variables", "occurs",
           "apply_subst", "substitute", "compose", "unify", "match",
           "factor_through", "unification"]

VAR = "VAR"
APP = "APP"


def var(name):
    r"""A variable term."""
    return (VAR, str(name))


def app(symbol, *args):
    r"""An application :math:`f(t_1, \ldots, t_n)`."""
    return (APP, str(symbol), tuple(args))


def const(symbol):
    r"""A constant: an application of arity zero."""
    return (APP, str(symbol), ())


def is_var(t):
    r"""Whether a term is a variable."""
    return isinstance(t, tuple) and len(t) == 2 and t[0] == VAR


def _check(t):
    if is_var(t):
        return t
    if (isinstance(t, tuple) and len(t) == 3 and t[0] == APP
            and isinstance(t[1], str)):
        return (APP, t[1], tuple(_check(a) for a in t[2]))
    raise ValueError("unifAlg: not a term: %r -- build terms with "
                     "var(), app() or const()" % (t,))


def variables(t):
    r"""Every variable name occurring in a term, in first-seen order."""
    out, seen = [], set()

    def walk(x):
        if is_var(x):
            if x[1] not in seen:
                seen.add(x[1])
                out.append(x[1])
        else:
            for a in x[2]:
                walk(a)

    walk(_check(t))
    return out


def occurs(name, t):
    r"""Whether variable ``name`` occurs anywhere inside ``t``."""
    return str(name) in variables(t)


def apply_subst(t, subst):
    r"""Apply a substitution to a term, repeatedly to a fixed point.

    The substitutions built here are idempotent, so one pass suffices;
    the loop guards against a hand-written substitution that is not.
    """
    cur = _check(t)
    for _ in range(64):
        nxt = _apply_once(cur, subst)
        if nxt == cur:
            return cur
        cur = nxt
    raise ValueError("unifAlg: the substitution %r does not reach a "
                     "fixed point -- it binds a variable to a term "
                     "containing itself" % (subst,))


def substitute(t, subst):
    r"""Apply a substitution in a single pass, without re-entering
    the terms it inserts.

    This is what a rewrite step needs: the bindings a matcher
    produces may mention the same variable names as the pattern, and
    iterating to a fixed point would then diverge on a substitution
    that is perfectly legitimate.
    """
    return _apply_once(_check(t), subst)


def _apply_once(t, subst):
    if is_var(t):
        return _check(subst[t[1]]) if t[1] in subst else t
    return (APP, t[1], tuple(_apply_once(a, subst) for a in t[2]))


def compose(outer, inner):
    r"""The substitution :math:`outer \circ inner`.

    Applying the result equals applying ``inner`` and then ``outer``.
    """
    out = {}
    for k, v in inner.items():
        out[k] = _apply_once(_check(v), outer)
    for k, v in outer.items():
        if k not in inner:
            out[k] = _check(v)
    return {k: v for k, v in out.items() if v != (VAR, k)}


def disagreement(t1, t2):
    r"""Robinson's disagreement pair: the leftmost position at which
    two terms differ, or ``None`` when they are identical.
    """
    a, b = _check(t1), _check(t2)
    if a == b:
        return None
    if is_var(a) or is_var(b):
        return (a, b)
    if a[1] != b[1] or len(a[2]) != len(b[2]):
        return (a, b)
    for x, y in zip(a[2], b[2]):
        d = disagreement(x, y)
        if d is not None:
            return d
    return None


def unify(t1, t2, occurs_check=True):
    r"""The most general unifier of two terms.

    Returns a :class:`RichResult` whose ``unified`` field says whether
    one exists and whose ``mgu`` field carries it. ``reason`` names
    the obstruction when there is none.
    """
    a, b = _check(t1), _check(t2)
    sub = {}
    for _ in range(4096):
        d = disagreement(apply_subst(a, sub), apply_subst(b, sub))
        if d is None:
            return RichResult(payload={
                "estimate": True, "unified": True, "mgu": sub,
                "reason": None, "occurs_check": bool(occurs_check),
                "cyclic": False, "n_bindings": len(sub),
                "method": "Robinson (1965) Sec. 5 disagreement-set "
                          "unification"})
        x, y = d
        if is_var(y) and not is_var(x):
            x, y = y, x
        if not is_var(x):
            why = ("symbol clash: %s/%d against %s/%d"
                   % (x[1], len(x[2]), y[1], len(y[2])))
            return _fail(sub, why, occurs_check)
        if occurs(x[1], y):
            if occurs_check:
                why = ("occurs check: %s occurs in the term it would "
                       "be bound to" % x[1])
                return _fail(sub, why, occurs_check)
            # No finite term satisfies this. Record the binding and
            # stop: pretending to continue would not terminate.
            cyc = dict(sub)
            cyc[x[1]] = y
            return RichResult(payload={
                "estimate": True, "unified": True, "mgu": cyc,
                "reason": None, "occurs_check": False, "cyclic": True,
                "n_bindings": len(cyc),
                "method": "Robinson (1965) Sec. 5 disagreement-set "
                          "unification, occurs check suppressed"})
        sub = compose({x[1]: y}, sub)
    raise ValueError("unifAlg: unification did not terminate")


def _fail(sub, why, oc):
    return RichResult(payload={
        "estimate": False, "unified": False, "mgu": None,
        "reason": why, "occurs_check": bool(oc),
        "partial": sub, "n_bindings": 0,
        "method": "Robinson (1965) Sec. 5 disagreement-set "
                  "unification"})


def match(pattern, subject):
    r"""One-way matching: a substitution on ``pattern`` alone.

    Unification treats both sides as open; matching holds ``subject``
    fixed, which is what a rewrite rule needs.
    """
    p, s = _check(pattern), _check(subject)
    sub = {}
    stack = [(p, s)]
    while stack:
        x, y = stack.pop()
        if is_var(x):
            if x[1] in sub:
                if sub[x[1]] != y:
                    return None
            else:
                sub[x[1]] = y
        elif is_var(y):
            return None
        elif x[1] != y[1] or len(x[2]) != len(y[2]):
            return None
        else:
            stack.extend(zip(x[2], y[2]))
    return sub


def factor_through(general, other, over):
    r"""The :math:`\delta` with ``other`` = :math:`\delta \circ`
    ``general`` on the variables ``over``, or ``None`` if none exists.

    This is what "most general" means, made checkable.
    """
    delta = {}
    for name in over:
        img = apply_subst(var(name), general)
        tgt = apply_subst(var(name), other)
        m = match(img, tgt)
        if m is None:
            return None
        for k, v in m.items():
            if k in delta and delta[k] != v:
                return None
            delta[k] = v
    return delta


# compact alias per ledger/NAMING.md
unification = unify
