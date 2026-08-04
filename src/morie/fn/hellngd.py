# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hellinger distance -- re-export of :mod:`morie.fn.hellie`.

The canonical implementation, with the source discussion, lives in
``morie.fn.hellie``; this module exists because the extraction pipeline
created two entries for the same method (Hellinger 1909).
"""

from __future__ import annotations

from .hellie import hellinger_distance

__all__ = ["hellinger_distance"]


def cheatsheet():
    return "hellngd(p, q): Hellinger distance (re-export of hellie)."
