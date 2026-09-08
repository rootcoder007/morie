# morie.fn -- function file (rootcoder007/morie)
"""Evaluation of an expression in prefix (Polish) notation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["prefixev", "prefix_evaluation"]


def prefixev(tokens):
    """Evaluate a Polish-notation expression, right to left.

    The whole point of the notation is that it needs no parentheses and
    no precedence table: scanning from the RIGHT, an operand is pushed
    and an operator pops exactly its arity.  Scanning left to right
    would require a lookahead the notation was designed to avoid.

    A malformed expression is caught rather than producing a plausible
    wrong number: the stack must hold exactly one value at the end, and
    an operator must find its operands.

    Formula: scan right to left; operand -> push;
             binary operator -> pop a, pop b, push (a op b)

    Parameters
    ----------
    tokens : sequence
        Tokens in prefix order.  Operators are the strings "+", "-",
        "*", "/", "^"; everything else must be a number.

    Returns
    -------
    RichResult
        ``value``, ``n_tokens``, ``n_operators``, ``max_stack``.

    References
    ----------
    Lukasiewicz, J. (1929/1963), Elements of Mathematical Logic, in
    which the parenthesis-free notation is introduced; the notation is
    usually dated to his 1924 lectures.  The evaluation procedure is
    the standard stack algorithm and is not attributed to that source.
    """
    toks = list(tokens)
    if not toks:
        raise ValueError("the expression is empty")
    ops = {"+": lambda a, b: a + b,
           "-": lambda a, b: a - b,
           "*": lambda a, b: a * b,
           "/": lambda a, b: a / b,
           "^": lambda a, b: a ** b}
    st = []
    nop = 0
    mx = 0
    for t in reversed(toks):
        if isinstance(t, str) and t in ops:
            if len(st) < 2:
                raise ValueError(
                    "operator '%s' has fewer than two operands" % t)
            a = st.pop()
            b = st.pop()
            if t == "/" and b == 0.0:
                raise ValueError("division by zero in the expression")
            st.append(ops[t](a, b))
            nop += 1
        else:
            st.append(float(t))
        if len(st) > mx:
            mx = len(st)
    if len(st) != 1:
        raise ValueError(
            "malformed prefix expression: %d values left on the stack"
            % len(st))
    return RichResult(payload={
        "value": st[0], "n_tokens": float(len(toks)),
        "n_operators": float(nop), "max_stack": float(mx),
        "method": "Prefix (Polish) notation evaluation"})


prefix_evaluation = prefixev


def cheatsheet():
    return "prdtA: scan right to left; operand pushes, operator pops two"
