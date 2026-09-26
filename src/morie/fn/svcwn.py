"""Condorcet winner test"""

# condorcet_winner was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .cndrc import condorcet_winner  # noqa: E402,F401

cond = condorcet_winner


def cheatsheet() -> str:
    return "condorcet_winner({}) -> Condorcet winner test"
