# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recursive character chunking (Alammar Ch 8; RAG)."""

from ._richresult import RichResult

__all__ = ["alammar_recursive_chunking"]


def alammar_recursive_chunking(text, separators=None, target_size=200,
                               overlap=0):
    """Split on the first separator; any piece still above target
    recurses with the NEXT separator; the last tier splits at exactly
    target_size characters. Overlap prepends the tail of the previous
    chunk. Reassembling the chunks (minus overlap) must give back the
    text, and the tests do exactly that.

    Examples
    --------
    >>> out = alammar_recursive_chunking("aa bb. cc dd. ee",
    ...     separators=[". ", " "], target_size=6)
    >>> out["chunks"]
    ['aa bb', 'cc dd', 'ee']
    """
    s = str(text)
    seps = list(separators) if separators is not None else ["\n\n", "\n",
                                                            ". ", " "]
    size = int(target_size)
    ov = int(overlap)
    if size < 1:
        raise ValueError("target_size must be positive.")
    if ov < 0 or ov >= size:
        raise ValueError("overlap must be non-negative and below "
                         "target_size.")

    def split(piece, tier):
        if len(piece) <= size:
            return [piece]
        if tier >= len(seps):
            return [piece[i:i + size] for i in range(0, len(piece), size)]
        parts = piece.split(seps[tier])
        if len(parts) == 1:
            return split(piece, tier + 1)
        out = []
        for p in parts:
            out.extend(split(p, tier + 1))
        return out

    chunks = [c for c in split(s, 0) if c]
    if ov > 0 and len(chunks) > 1:
        with_ov = [chunks[0]]
        for i in range(1, len(chunks)):
            with_ov.append(chunks[i - 1][-ov:] + chunks[i])
        chunks = with_ov
    return RichResult(payload={
        "chunks": chunks, "n_chunks": len(chunks),
        "max_chunk_length": max(len(c) for c in chunks) if chunks else 0,
        "overlap": ov, "estimate": float(len(chunks)), "n": len(s),
        "method": "Recursive character chunking (Alammar Ch 8)"})


def cheatsheet():
    return "alchk: tiered separator splitting with hard fallback and overlap"
