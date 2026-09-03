"""RNA covariance models: covariation in a structural alignment.

RNA structure is held together by base pairs, and base pairs leave a
signature in an alignment that sequence conservation does not. If
position 12 pairs with position 40, then a mutation at 12 that would
break the pair is only tolerated when position 40 mutates to restore
it -- G:C becomes A:U, never G:U-and-stay. So the two columns VARY
TOGETHER while each varies freely on its own.

That is the whole idea, and the thing it corrects is the intuition that
conserved means important. A column pair that is G:C in every single
sequence tells you nothing about whether the two positions interact:
they might be conserved for entirely separate reasons. Covariation is
evidence of pairing; conservation is not. Mutual information says
exactly this and says it in bits:

    I(i, j) = sum_{a,b} f_ij(a,b) log2[ f_ij(a,b) / (f_i(a) f_j(b)) ]

which is zero when the columns are independent AND zero when either is
constant -- the second being the case the eye gets wrong.

Raw mutual information has a known bias: with few sequences, two columns
look dependent by chance, and the bias grows with the number of letters
actually seen. The Miller-Madow correction subtracts the leading term of
that bias, and it is offered as a route rather than applied silently,
because a corrected and an uncorrected score are different numbers and
the reader should know which they have.

Two structure routes:

  "given"     score the base pairs of a consensus structure supplied in
              dot-bracket notation.
  "nussinov"  fold each column pair set by maximum base pairing first,
              then score. This is the classic dynamic program and it is
              here as the structure-free alternative -- it knows nothing
              about covariation, so comparing the two says whether the
              proposed structure is doing better than pairing counts
              alone.

Gaps are their own problem. A column pair whose sequences are mostly
gapped has almost no data behind it, and the effective number of
ungapped sequences per pair is reported so a high score on four
sequences cannot be mistaken for a high score on four hundred.

References
  Eddy, S.R. and Durbin, R. (1994) "RNA sequence analysis using
    covariance models." Nucleic Acids Research 22(11), 2079-2088.
    doi:10.1093/nar/22.11.2079. Covariance models and the mutual
    information signal.
  Nawrocki, E.P. and Eddy, S.R. (2013) "Infernal 1.1: 100-fold faster
    RNA homology searches." Bioinformatics 29(22), 2933-2935. The
    modern implementation of the same model.
  Rivas, E., Clements, J. and Eddy, S.R. (2017) "A statistical test for
    conserved RNA structure shows lack of evidence for structure in
    lncRNAs." Nature Methods 14(1), 45-48. Why covariation and not
    conservation is the evidence.
  Nussinov, R. and Jacobson, A.B. (1980) "Fast algorithm for predicting
    the secondary structure of single-stranded RNA." PNAS 77(11),
    6309-6313. The maximum-pairing dynamic program.
  Miller, G.A. (1955) "Note on the bias of information estimates." In
    Information Theory in Psychology, 95-100. The bias correction.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["rnacov", "rna_covariance", "mutual_information",
           "parse_structure", "nussinov", "column_counts", "ALPHABET",
           "PAIRS", "STRUCTURES", "cheatsheet"]

ALPHABET = ("A", "C", "G", "U")
# Watson-Crick plus the wobble, which is a real pair and leaving it out
# would score two thirds of a stem as unpaired.
PAIRS = (("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"),
         ("U", "G"))
STRUCTURES = ("given", "nussinov")

_GAPS = ("-", ".", "~")


def column_counts(alignment, i, j):
    """Joint and marginal letter counts for two columns, gaps dropped.

    A sequence gapped in either column contributes to neither margin.
    Counting it in one and not the other would make the marginals and
    the joint disagree about how many sequences there were, and the
    mutual information would stop being a mutual information.
    """
    idx = {}
    for t, a in enumerate(ALPHABET):
        idx[a] = t
    joint = [[0] * 4 for _ in range(4)]
    n = 0
    for s in alignment:
        a = s[i]
        b = s[j]
        if a in _GAPS or b in _GAPS or a not in idx or b not in idx:
            continue
        joint[idx[a]][idx[b]] += 1
        n += 1
    mi = [0] * 4
    mj = [0] * 4
    for a in range(4):
        for b in range(4):
            mi[a] += joint[a][b]
            mj[b] += joint[a][b]
    return joint, mi, mj, n


def mutual_information(alignment, i, j, correction="none"):
    """Mutual information between two alignment columns, in bits.

    Zero for independent columns and zero when either column is
    constant. The second is the point: a perfectly conserved pair is
    perfectly uninformative about whether the positions interact.
    """
    joint, mi, mj, n = column_counts(alignment, i, j)
    if n == 0:
        return 0.0, 0, 0
    terms = []
    seen = 0
    for a in range(4):
        for b in range(4):
            c = joint[a][b]
            if c == 0:
                continue
            seen += 1
            pab = c / float(n)
            pa = mi[a] / float(n)
            pb = mj[b] / float(n)
            terms.append(pab * math.log(pab / (pa * pb)) / math.log(2.0))
    v = _w.csum(terms) if terms else 0.0
    if correction == "miller_madow":
        # The leading bias term: (cells seen - rows seen - cols seen + 1)
        # over 2 n ln 2. It is subtracted, so a small sample's spurious
        # dependence is charged for rather than reported as signal.
        rows = sum(1 for a in range(4) if mi[a] > 0)
        cols = sum(1 for b in range(4) if mj[b] > 0)
        v = v - (seen - rows - cols + 1) / (2.0 * n * math.log(2.0))
    elif correction != "none":
        raise ValueError("correction must be none or miller_madow")
    return v, n, seen


def parse_structure(s):
    """Dot-bracket to a list of base pairs.

    An unbalanced string is an error, not a structure. Silently dropping
    an unmatched bracket would give a plausible-looking pair list for a
    string that never described a structure at all.
    """
    stack = []
    pairs = []
    for k, ch in enumerate(s):
        if ch in "(<[{":
            stack.append(k)
        elif ch in ")>]}":
            if not stack:
                raise ValueError("closing bracket at %d has nothing to "
                                 "close" % k)
            pairs.append((stack.pop(), k))
        elif ch not in ".:_-,":
            raise ValueError("character %r at %d is not dot-bracket"
                             % (ch, k))
    if stack:
        raise ValueError("%d bracket(s) never closed, first at %d"
                         % (len(stack), stack[0]))
    pairs.sort()
    return pairs


def _can_pair(a, b):
    for x, y in PAIRS:
        if a == x and b == y:
            return True
    return False


def nussinov(seq, min_loop=3):
    """Maximum base pairing by the Nussinov dynamic program.

    Fills the upper triangle with the best number of pairs on each
    subsequence, then traces back. The minimum loop length is a physical
    constraint, not a tuning knob: a hairpin cannot close on fewer than
    about three unpaired bases.
    """
    n = len(seq)
    m = [[0] * n for _ in range(n)]
    for span in range(min_loop + 1, n):
        for i in range(n - span):
            j = i + span
            best = m[i][j - 1]
            if _can_pair(seq[i], seq[j]):
                inner = m[i + 1][j - 1] if i + 1 <= j - 1 else 0
                if inner + 1 > best:
                    best = inner + 1
            for k in range(i, j):
                v = m[i][k] + m[k + 1][j]
                if v > best:
                    best = v
            m[i][j] = best
    pairs = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i <= min_loop:
            continue
        if m[i][j] == m[i][j - 1]:
            stack.append((i, j - 1))
            continue
        if _can_pair(seq[i], seq[j]):
            inner = m[i + 1][j - 1] if i + 1 <= j - 1 else 0
            if m[i][j] == inner + 1:
                pairs.append((i, j))
                stack.append((i + 1, j - 1))
                continue
        for k in range(i, j):
            if m[i][j] == m[i][k] + m[k + 1][j]:
                stack.append((i, k))
                stack.append((k + 1, j))
                break
    pairs.sort()
    return pairs, m[0][n - 1] if n else 0


def rna_covariance(alignment, structure=None, correction="none",
                   mode="given", min_loop=3, min_sequences=4):
    """Score the covariation supporting a structure in an alignment.

    Parameters
    ----------
    alignment : sequence of str
        Aligned sequences, all the same length.
    structure : str or None
        Consensus structure in dot-bracket notation, for the "given"
        mode.
    correction : str
        "none" or "miller_madow".
    mode : str
        A member of STRUCTURES.
    min_sequences : int
        Pairs with fewer ungapped sequences than this are reported as
        unsupported rather than scored as if they were.

    Returns
    -------
    RichResult
        Per-pair mutual information and support, the total, the pairs
        used, and how many were too sparsely covered to judge.

    References
    ----------
    Eddy and Durbin (1994) Nucleic Acids Res 22(11), 2079-2088; Rivas,
    Clements and Eddy (2017) Nat Methods 14(1), 45-48.
    """
    if mode not in STRUCTURES:
        raise ValueError("mode must be one of %r" % (STRUCTURES,))
    seqs = [str(s).upper().replace("T", "U") for s in alignment]
    if not seqs:
        raise ValueError("the alignment is empty")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("every sequence must have the same length")

    if mode == "given":
        if structure is None:
            raise ValueError("the given mode needs a structure")
        pairs = parse_structure(structure)
        folded = 0
    else:
        # Fold the first ungapped sequence: the dynamic program works on
        # a sequence, not an alignment, and using the first one is a
        # stated choice rather than a silent consensus nobody defined.
        base = seqs[0].replace("-", "").replace(".", "")
        pairs, folded = nussinov(base, min_loop)

    for i, j in pairs:
        if i < 0 or j >= L or i >= j:
            raise ValueError("pair (%d, %d) is outside the alignment"
                             % (i, j))

    mis = []
    sup = []
    cells = []
    weak = 0
    for i, j in pairs:
        v, n, seen = mutual_information(seqs, i, j, correction)
        mis.append(v)
        sup.append(n)
        cells.append(seen)
        if n < int(min_sequences):
            weak += 1
    total = _w.csum(mis) if mis else 0.0
    strong = [k for k in range(len(pairs))
              if sup[k] >= int(min_sequences) and mis[k] > 0.0]
    return RichResult(payload={
        "pair_i": [p[0] for p in pairs],
        "pair_j": [p[1] for p in pairs],
        "mutual_information": mis,
        "support": sup,
        "cells_seen": cells,
        "n_pairs": len(pairs),
        "n_weak": weak,
        "n_covarying": len(strong),
        "covarying": strong,
        "total": total,
        "estimate": total / len(pairs) if pairs else float("nan"),
        "se": float("nan"),
        "max_mi": max(mis) if mis else float("nan"),
        "n_sequences": len(seqs),
        "length": L,
        "folded_pairs": folded,
        "correction": correction,
        "mode": mode,
        "method": "RNA covariance model scoring",
    })


rnacov = rna_covariance


def cheatsheet():
    return ("rnacov: RNA covariance model scoring. modes "
            + ", ".join(STRUCTURES)
            + "; mutual information in bits, conservation is not "
              "covariation")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
rnacovariance = rna_covariance
