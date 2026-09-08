# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tokenisation pipeline: normalise, pre-tokenise, subword,
post-process (Alammar Ch 2)."""

from ._richresult import RichResult

__all__ = ["alammar_tokenization_pipeline"]


def alammar_tokenization_pipeline(text, vocab, unk_token="[UNK]",
                                  lowercase=True, specials=("[CLS]",
                                                            "[SEP]")):
    """Post(Subword(Pre(Normalise(text)))): lowercase + whitespace
    split + greedy longest-match WordPiece ("##" continuations) +
    special-token wrapping.

    Every emitted token is either in the vocabulary or the UNK token,
    and the tests assert that invariant over fuzzed input.

    Examples
    --------
    >>> v = ["[CLS]", "[SEP]", "[UNK]", "un", "##happy", "dog"]
    >>> alammar_tokenization_pipeline("Unhappy dog", v)["tokens"]
    ['[CLS]', 'un', '##happy', 'dog', '[SEP]']
    """
    voc = [str(v) for v in vocab]
    vs = set(voc)
    if unk_token not in vs:
        raise ValueError(f"the vocabulary must contain {unk_token!r}.")
    s = str(text)
    if lowercase:
        s = s.lower()
    words = s.split()
    toks = []
    for w in words:
        i = 0
        pieces = []
        ok = True
        while i < len(w):
            j = len(w)
            found = None
            while j > i:
                cand = w[i:j] if i == 0 else "##" + w[i:j]
                if cand in vs:
                    found = cand
                    break
                j -= 1
            if found is None:
                ok = False
                break
            pieces.append(found)
            i = j
        toks.extend(pieces if ok else [unk_token])
    out = [specials[0], *toks, specials[1]] if specials else toks
    missing = [t for t in (specials or ()) if t not in vs]
    if missing:
        raise ValueError(f"special tokens {missing} are not in the "
                         "vocabulary.")
    return RichResult(payload={
        "tokens": out, "n_unk": out.count(unk_token),
        "estimate": float(len(out)), "n": len(words),
        "method": "WordPiece tokenisation pipeline (Alammar Ch 2)"})


def cheatsheet():
    return "altkp: normalise, split, greedy WordPiece, wrap specials"
