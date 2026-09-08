# morie.fn -- function file (rootcoder007/morie)
r"""Parsing expression grammars: no ambiguity to resolve.

Context-free grammars were built to model *natural* language, where
ambiguity is real and must be represented. Machine-oriented languages
have no such need, and inheriting the machinery has a cost: the
dangling-else, the need to prove a grammar unambiguous, the split
between a lexer and a parser, and parsers that are either fickle (LR)
or slow (generalised CFG parsing).

**PEGs remove ambiguity by never introducing it.** Where a CFG's
:math:`e_1 \mid e_2` is a non-deterministic choice, a PEG's
:math:`e_1\,/\,e_2` is **prioritised**: try :math:`e_1`; only if it
fails, try :math:`e_2` from the same position. A PEG therefore
describes a *recognition* procedure rather than a generative system,
and it always has at most one parse.

**That single change has consequences worth stating plainly.**

* ``A <- "a" / "ab"`` never matches ``"ab"`` fully -- the first
  alternative succeeds on ``"a"`` and the second is never tried. In a
  CFG both derivations exist. This is the classic PEG surprise, and
  the anchor pins it.
* Repetition and option are **greedy**, not non-deterministic: ``e*``
  consumes as much as it can and does not give any back.
* Sequence backtracks to the start of the sequence if either half
  fails, but the *choice* commits once an alternative succeeds. Local
  backtracking, not global search.
* Syntactic predicates ``&e`` and ``!e`` test without consuming, which
  is what gives unlimited lookahead and lets lexical and hierarchical
  syntax live in one grammar.

**Linear time is available for any PEG** by memoising each
(rule, position) result -- the packrat technique -- trading memory for
the guarantee. Both the plain and memoised recognisers are implemented
so the difference in work is measurable rather than asserted.

References
----------
Ford, B. (2004) "Parsing Expression Grammars: A Recognition-Based
Syntactic Foundation", *Proceedings of the 31st ACM SIGPLAN-SIGACT
Symposium on Principles of Programming Languages (POPL '04)*, 111-122,
doi:10.1145/964001.964011. The abstract and Sec. 1 (CFGs' power to
express ambiguity serves natural language and makes machine-oriented
syntax unnecessarily hard; PEGs solve ambiguity by not introducing it;
prioritised choice in place of non-deterministic choice; no need to
separate lexical and hierarchical components; a linear-time parser
exists for any PEG, avoiding both LR's fickleness and the inefficiency
of generalised CFG parsing; reducibility to TS/TDPL and gTS/GTDPL).
Sec. 3 (the operators: sequence e1 e2 backtracking to the starting
point if either fails, the choice e1 / e2 attempting e2 from the same
point only if e1 fails, and ?, * and + being greedy rather than
non-deterministic).

Ford, B. (2002) "Packrat Parsing: Simple, Powerful, Lazy, Linear
Time", *ICFP 2002*, 36-47, doi:10.1145/581478.581483. The memoisation
that gives the linear-time guarantee.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["lit", "seq", "choice", "star", "plus", "opt", "and_",
           "not_", "parse", "packrat_parse"]

_EPS = 1e-12
FAIL = None


def lit(s):
    r"""A terminal string."""
    def f(text, pos, ctx):
        ctx["steps"] += 1
        if text.startswith(s, pos):
            return pos + len(s)
        return FAIL
    f.tag = ("lit", s)
    return f


def seq(*es):
    r""":math:`e_1 e_2`: backtrack to the starting point if either
    fails."""
    def f(text, pos, ctx):
        ctx["steps"] += 1
        p = pos
        for e in es:
            p = e(text, p, ctx)
            if p is FAIL:
                return FAIL
        return p
    f.tag = ("seq", len(es))
    return f


def choice(*es):
    r""":math:`e_1 / e_2`: PRIORITISED -- once one succeeds the rest
    are never tried."""
    def f(text, pos, ctx):
        ctx["steps"] += 1
        for e in es:
            p = e(text, pos, ctx)
            if p is not FAIL:
                return p
        return FAIL
    f.tag = ("choice", len(es))
    return f


def star(e):
    r""":math:`e^*`: GREEDY, and gives nothing back."""
    def f(text, pos, ctx):
        ctx["steps"] += 1
        p = pos
        while True:
            q = e(text, p, ctx)
            if q is FAIL or q == p:
                return p
            p = q
    f.tag = ("star",)
    return f


def plus(e):
    return seq(e, star(e))


def opt(e):
    r""":math:`e?`: unconditionally consumes what :math:`e` matched."""
    return choice(e, lit(""))


def and_(e):
    r""":math:`\&e`: succeeds without consuming."""
    def f(text, pos, ctx):
        ctx["steps"] += 1
        return pos if e(text, pos, ctx) is not FAIL else FAIL
    f.tag = ("and",)
    return f


def not_(e):
    r""":math:`!e`: succeeds when :math:`e` fails, consuming nothing.

    With ``and_`` this gives unlimited lookahead, which is why one
    grammar can cover lexical and hierarchical syntax.
    """
    def f(text, pos, ctx):
        ctx["steps"] += 1
        return pos if e(text, pos, ctx) is FAIL else FAIL
    f.tag = ("not",)
    return f


def parse(expr, text, full=True):
    r"""Recognise ``text``; ``full`` requires the whole input."""
    ctx = {"steps": 0, "memo": None}
    end = expr(str(text), 0, ctx)
    ok = end is not FAIL and (not full or end == len(str(text)))
    return RichResult(payload={
        "estimate": ok, "matched": ok,
        "end": end, "consumed": 0 if end is FAIL else end,
        "steps": ctx["steps"], "memoised": False,
        "method": "PEG recognition; Ford (2004)",
        "note": "prioritised choice, so there is at most ONE parse",
    })


def packrat_parse(expr, text, full=True):
    r"""The same, memoising (rule, position) -- linear time for any
    PEG."""
    ctx = {"steps": 0, "memo": {}}
    memo = ctx["memo"]

    def wrap(e):
        key = getattr(e, "tag", None)

        def f(t, pos, c):
            kk = (id(e), pos)
            if kk in memo:
                return memo[kk]
            r = e(t, pos, c)
            memo[kk] = r
            return r
        f.tag = key
        return f

    end = wrap(expr)(str(text), 0, ctx)
    ok = end is not FAIL and (not full or end == len(str(text)))
    return RichResult(payload={
        "estimate": ok, "matched": ok, "end": end,
        "steps": ctx["steps"], "memo_entries": len(memo),
        "memoised": True,
        "method": "packrat parsing; Ford (2002)",
    })


def cheatsheet():
    return ("prsPEG: CFG ambiguity exists to model NATURAL language "
            "and only costs you in machine syntax. A PEG's e1 / e2 is "
            "PRIORITISED, not non-deterministic, so there is at most "
            "one parse and nothing to disambiguate. Consequence: "
            "A <- 'a' / 'ab' never matches 'ab' -- the first "
            "alternative wins and the second is never tried. *, + and "
            "? are GREEDY and give nothing back; sequence backtracks, "
            "choice commits. &e and !e look ahead without consuming, "
            "so one grammar covers lexing and parsing. Memoise "
            "(rule, position) for linear time.")


# compact alias per ledger/NAMING.md
parsingexpressiongrammar = parse

# public names resolved by fn/_lazy_map.json
peg_parser = parse
pegparser = parse
