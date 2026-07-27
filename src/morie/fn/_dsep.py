# morie.fn -- function file (rootcoder007/morie)
"""Shared d-separation helper on top of the bdcrt path machinery."""

from .bdcrt import _blocked, _descendants, _has_cycle, _parse, _paths

__all__ = ["d_separated"]


def d_separated(dag, X, Y, Z=()):
    """True when X and Y are d-separated given Z in the DAG."""
    children, parents, nodes = _parse(dag)
    if _has_cycle(children, nodes):
        raise ValueError("dag contains a directed cycle.")
    for n in (X, Y, *Z):
        if n not in nodes:
            raise ValueError(f"node {n!r} not in the graph.")
    Z = set(Z)
    return all(_blocked(path, dirs, Z, children) for path, dirs in _paths(X, Y, children, parents))


def cheatsheet():
    return "_dsep: d_separated(dag, X, Y, Z) via bdcrt path enumeration"
