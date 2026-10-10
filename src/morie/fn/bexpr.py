# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Evaluate a boolean expression given variable assignments."""

from __future__ import annotations

from ._containers import DescriptiveResult


def boolean_eval(
    expression: str,
    variables: dict[str, int],
) -> DescriptiveResult:
    """
    Evaluate a boolean expression given variable assignments.

    Supports operators: AND (&), OR (|), NOT (~), XOR (^).
    Variables are single uppercase letters.

    :param expression: Boolean expression string (e.g. "A & (B | ~C)").
    :param variables: Dict mapping variable names to 0 or 1.
    :return: DescriptiveResult with evaluation result.
    :raises ValueError: If expression contains undefined variables.

    References
    ----------
    Boole, G. (1854). *An Investigation of the Laws of Thought*.
    Walton and Maberly.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression must be non-empty.")

    for var, val in variables.items():
        if val not in (0, 1):
            raise ValueError(f"Variable {var} must be 0 or 1, got {val}.")

    import re

    def _value(m):
        name = m.group(0)
        if name in ("True", "False", "and", "or", "not"):  # literals and word operators stay as written
            return name
        if name not in variables:
            raise ValueError(f"Undefined variable: {name}")
        return str(bool(variables[name]))

    # whole identifiers only: a text replace of a lower-case name ("o", "n") also rewrote the
    # letters of the True/False it had just inserted
    expr = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _value, expression)

    expr = expr.replace("~", " not ")
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " != ")
    expr = expr.strip()  # "~A" became " not True": the expression parser rejects a leading space

    allowed = set("TrueFalsendorat!=() 01")
    cleaned = expr.replace("not", "").replace("and", "").replace("or", "")
    for ch in cleaned:
        if ch not in allowed and not ch.isspace():
            raise ValueError(f"Unexpected character in expression: '{ch}'")

    # AST-validated evaluation (boolean/comparison operators and literals
    # only, no builtins reachable) -- in place of handing the text to the interpreter.
    from morie._safe_expr import safe_eval_expr

    result = int(bool(safe_eval_expr(expr)))

    return DescriptiveResult(
        name="Boolean Evaluation",
        value=result,
        extra={
            "expression": expression,
            "variables": variables,
            "result_bool": bool(result),
        },
    )


short = boolean_eval


def cheatsheet() -> str:
    return "boolean_eval({}) -> Boolean expression evaluator."


# compact alias per ledger/NAMING.md
booleaneval = boolean_eval
