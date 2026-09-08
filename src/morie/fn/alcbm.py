# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conversation buffer memory of the last N turns (Alammar Ch 7)."""

from ._richresult import RichResult

__all__ = ["alammar_conversation_buffer_memory"]


def alammar_conversation_buffer_memory(conversation, N):
    """memory = the last N (user, assistant) turns, oldest first.

    What was DROPPED is reported: buffer memory's failure mode is
    silent amnesia, and the count of forgotten turns is the honest
    number to surface.

    Examples
    --------
    >>> out = alammar_conversation_buffer_memory(
    ...     [("u1", "a1"), ("u2", "a2"), ("u3", "a3")], 2)
    >>> out["memory"]
    [('u2', 'a2'), ('u3', 'a3')]
    >>> out["turns_forgotten"]
    1
    """
    turns = [(str(u), str(a)) for (u, a) in conversation]
    n = int(N)
    if n < 1:
        raise ValueError("N must be positive.")
    kept = turns[-n:]
    return RichResult(payload={
        "memory": kept, "turns_forgotten": max(0, len(turns) - n),
        "estimate": float(len(kept)), "n": len(turns),
        "method": "Conversation buffer memory (Alammar Ch 7)"})


def cheatsheet():
    return "alcbm: last-N turn window, forgotten turns counted"
