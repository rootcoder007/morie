# morie.fn -- slice s03 (rootcoder007/morie)
"""Canonical serialisation of AlphaZero training data.

There is no equation here; Silver et al. (2018), arXiv:1712.01815
(FETCHED), says only that the (s, pi, z) triples are stored and sampled
uniformly from the most recent games.  What this function contributes is
the one property that matters for a training pipeline and is checkable:
a *canonical* encoding, so that the same buffer always produces the same
bytes and the same digest regardless of dictionary ordering or of which
language wrote it.

The digest is a Rabin-Karp polynomial hash over the canonical text,

    h <- (131 h + c) mod (2^31 - 1)

which is exact in double precision because every intermediate stays
below 2^53.  It is not a cryptographic hash and is not presented as
one; it is an integrity check on the record.

Writing to disk is opt-in (``path``); by default nothing touches the
filesystem, which keeps the function pure and testable.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_data_pickle"]

_MOD = 2147483647.0
_BASE = 131.0


def _digest(text):
    h = 0.0
    for ch in text:
        h = (_BASE * h + float(ord(ch))) % _MOD
    return h


def _fmt(x):
    return "%.17g" % float(x)


def alphazero_data_pickle(replay_buffer, path=None):
    """Encode a replay buffer canonically and digest it.

    Parameters
    ----------
    replay_buffer : array-like
        Rows of (s, pi..., z), or any nested numeric structure; each row
        is flattened.
    path : str, optional
        When given, the canonical text is written there.

    Returns
    -------
    RichResult with payload:
        estimate : the digest
        digest   : same as estimate
        n_rows, n_values, text_len
        written  : whether a file was written
    """
    rows = k.mat(replay_buffer)
    parts = []
    nv = 0
    for r in rows:
        parts.append(",".join([_fmt(x) for x in r]))
        nv += len(r)
    text = "\n".join(parts)
    written = False
    if path is not None:
        with open(path, "w") as fh:
            fh.write(text)
        written = True
    return RichResult(
        title="AlphaZero training data",
        summary_lines=[("rows", len(rows)), ("digest", _digest(text))],
        payload={
            "estimate": _digest(text),
            "digest": _digest(text),
            "n_rows": len(rows),
            "n_values": nv,
            "text_len": len(text),
            "written": written,
            "method": "Canonical (s, pi, z) encoding with a Rabin-Karp digest",
        },
    )


def cheatsheet():
    return "agdpck: AlphaZero training data serialization"
