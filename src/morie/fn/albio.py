# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BIO / BIOES span tagging (Alammar Ch 4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_bio_tagging"]


def alammar_bio_tagging(tokens, entity_spans, scheme="BIO"):
    """Tags from (start, end, type) spans, end exclusive.

    Overlapping spans are refused: BIO cannot represent them, and
    letting the later span silently overwrite the earlier one is a
    data corruption, not a convention.

    Examples
    --------
    >>> alammar_bio_tagging(["a", "b", "c"], [(0, 2, "PER")])["tags"]
    ['B-PER', 'I-PER', 'O']
    >>> alammar_bio_tagging(["a", "b", "c"], [(0, 2, "PER")],
    ...                     scheme="BIOES")["tags"]
    ['B-PER', 'E-PER', 'O']
    """
    toks = [str(t) for t in np.atleast_1d(np.asarray(tokens, dtype=object))]
    n = len(toks)
    scheme = str(scheme).upper()
    if scheme not in ("BIO", "BIOES"):
        raise ValueError(f"scheme must be BIO or BIOES; got {scheme!r}.")
    tags = ["O"] * n
    claimed = [False] * n
    for (s, e, typ) in entity_spans:
        s, e = int(s), int(e)
        if not (0 <= s < e <= n):
            raise ValueError(
                f"span ({s}, {e}) is out of range for {n} tokens "
                "(end exclusive).")
        if any(claimed[s:e]):
            raise ValueError(
                f"span ({s}, {e}) overlaps an earlier span; BIO cannot "
                "represent overlapping entities.")
        for i in range(s, e):
            claimed[i] = True
        if scheme == "BIO":
            tags[s] = f"B-{typ}"
            for i in range(s + 1, e):
                tags[i] = f"I-{typ}"
        else:
            if e - s == 1:
                tags[s] = f"S-{typ}"
            else:
                tags[s] = f"B-{typ}"
                for i in range(s + 1, e - 1):
                    tags[i] = f"I-{typ}"
                tags[e - 1] = f"E-{typ}"
    return RichResult(payload={
        "tags": tags, "n_entities": len(list(entity_spans)),
        "estimate": float(sum(t != "O" for t in tags)), "n": n,
        "method": f"{scheme} span tagging (Alammar Ch 4)"})


def cheatsheet():
    return "albio: B/I/O or B/I/O/E/S tags, overlaps refused"
