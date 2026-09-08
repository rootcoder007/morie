# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FastText word representation: sum of subword n-gram embeddings."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_fasttext_subword", "word_ngrams"]


def word_ngrams(word, n_min, n_max, boundary="<>"):
    """The character n-grams of a boundary-marked word, in order, plus
    the marked whole word (FastText's special sequence). Duplicated
    n-grams are kept: a word containing the same n-gram twice sums
    that vector twice, which is what the formula says."""
    n_min, n_max = int(n_min), int(n_max)
    if n_min < 1 or n_max < n_min:
        raise ValueError(f"need 1 <= n_min <= n_max; got {n_min}, {n_max}.")
    if not word:
        raise ValueError("the empty word has no n-grams.")
    marked = boundary[0] + str(word) + boundary[1]
    grams = [marked[i:i + n] for n in range(n_min, n_max + 1)
             for i in range(len(marked) - n + 1)]
    if marked not in grams:
        grams.append(marked)
    return grams


def kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max):
    """v_w = sum_{g in Ngrams(w)} z_g.

    ``ngram_embeddings`` maps an n-gram string to its vector. An
    n-gram absent from the table contributes nothing but is COUNTED
    and reported -- a word whose subwords are all unknown returns a
    zero vector by arithmetic, and pretending otherwise hides an
    out-of-vocabulary tokenizer.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 1,
    FastText; the section is not present in the 2024 PDF, so the sum
    is implemented exactly as the spec line states (Bojanowski et al.
    2017 subword model).

    Examples
    --------
    >>> tbl = {"<a": [1.0, 0.0], "ab": [0.0, 1.0], "b>": [1.0, 1.0],
    ...        "<ab>": [0.0, 0.0]}
    >>> out = kamath_fasttext_subword("ab", tbl, 2, 2)
    >>> out["vector"]
    [2.0, 2.0]
    >>> out["n_known"], out["n_missing"]
    (4, 0)
    """
    grams = word_ngrams(word, n_min, n_max)
    if not isinstance(ngram_embeddings, dict):
        raise ValueError("ngram_embeddings must be a dict n-gram -> vector.")
    if not ngram_embeddings:
        raise ValueError("ngram_embeddings is empty.")
    dim = None
    total = None
    known, missing = 0, []
    for g in grams:
        z = ngram_embeddings.get(g)
        if z is None:
            missing.append(g)
            continue
        z = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
        if dim is None:
            dim, total = z.size, z.copy()
        else:
            if z.size != dim:
                raise ValueError(
                    f"n-gram {g!r} has width {z.size}, expected {dim}.")
            total = total + z
        known += 1
    if total is None:
        raise ValueError(
            f"none of the {len(grams)} n-grams of {word!r} is in the "
            "table, so the word has no representation at all.")
    return RichResult(payload={
        "vector": [float(v) for v in total],
        "estimate": float(total[0]),
        "ngrams": grams, "n_known": known,
        "n_missing": len(missing), "missing": missing,
        "n": len(grams),
        "method": "FastText subword sum v_w = sum z_g"})


def cheatsheet():
    return "kmfst: v_w = sum of the char n-gram vectors of <word>"


# compact alias per ledger/NAMING.md
wordngrams = word_ngrams
