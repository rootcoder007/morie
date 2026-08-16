"""The OpenFold MSA-pair head: the two directions in which an Evoformer
block lets a multiple sequence alignment and a pairwise representation
speak to each other.

An alignment is a stack of sequences; a pair representation is a matrix
over residue positions. Neither is much use alone. The alignment knows
which positions covary across evolution and nothing about geometry; the
pair representation carries the geometry and has no memory of the
alignment that implied it. The head is the pipe between them, and it
runs both ways in every block:

  MSA -> pair, the OUTER PRODUCT MEAN (AlphaFold 2 Supplementary
  Algorithm 10). For each ordered pair of positions (i, j), take the
  outer product of their channel vectors within one sequence and average
  it over the sequences of the alignment:

      o_ij[a][b] = (1/S) sum_s  m[s][i][a] * m[s][j][b]

  That average is the whole idea. A single sequence contributes a rank
  one matrix that says nothing; it is the variation ACROSS the alignment
  that makes o_ij informative, which is why the operation is a mean over
  sequences and not a sum over anything else.

  pair -> MSA, ROW-WISE GATED SELF-ATTENTION WITH A PAIR BIAS
  (Supplementary Algorithm 7). Within one sequence, position i attends
  over positions j, and the pair representation enters as an additive
  bias on the attention logit:

      logit_ij = (q_i . k_j) / sqrt(c) + b_ij

  The bias is the only channel through which geometry reaches the
  alignment, so it is the part worth being able to test: setting b to
  zero must change the attention, and a large b at one column must pull
  the attention there.

WHAT IS HERE AND WHAT IS NOT. Both operations above are pure tensor
algebra and are implemented exactly. Everything in an Evoformer block
that is a TRAINED WEIGHT -- the query, key and value projections, the
sigmoid gate, the linear that turns a pair vector into a scalar bias,
and the linear that projects the c*c outer product back down to the
pair width -- is a parameter of this function, not a constant of it.
The published papers give the architecture; the weights are a download,
not a formula, and this module does not pretend to know them. Every one
of them defaults to the identity or to absence, and each default is
named at its parameter. A caller holding real OpenFold weights passes
them in and gets the real head.

References
  Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M.,
    Ronneberger, O., Tunyasuvunakool, K., Bates, R., Zidek, A.,
    Potapenko, A., Bridgland, A., Meyer, C., Kohl, S.A.A., Ballard,
    A.J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler,
    J., Back, T., Petersen, S., Reiman, D., Clancy, E., Zielinski, M.,
    Steinegger, M., Pacholska, M., Berghammer, T., Bodenstein, S.,
    Silver, D., Vinyals, O., Senior, A.W., Kavukcuoglu, K., Kohli, P.
    and Hassabis, D. (2021) "Highly accurate protein structure
    prediction with AlphaFold." Nature 596, 583-589.
    doi:10.1038/s41586-021-03819-2. Supplementary Algorithms 7 and 10
    are the two operations implemented here.
  Ahdritz, G., Bouatta, N., Floristean, C., Kadyan, S., Xia, Q.,
    Gerecke, W., O'Donnell, T.J., Berenberg, D., Fisk, I., Zanichelli,
    N., Zhang, B., Nowaczynski, A., Wang, B., Stepniewska-Dziubinska,
    M.M., Zhang, S., Ojewole, A., Guney, M.E., Biderman, S., Watkins,
    A.M., Ra, S., Lorenzo, P.R., Nivon, L., Weitzner, B., Ban, Y.A.,
    Sorger, P.K., Mostaque, E., Zhang, Z., Bonneau, R. and AlQuraishi,
    M. (2022) "OpenFold: retraining AlphaFold2 yields new insights into
    its learning mechanisms and capacity for generalization." bioRxiv
    2022.11.20.517210. doi:10.1101/2022.11.20.517210.
  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
    Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) "Attention is all
    you need." Advances in Neural Information Processing Systems 30,
    5998-6008. The scaled dot-product attention the bias is added to.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["openfold_msa_pair", "outer_product_mean", "pair_bias",
           "msa_row_attention", "softmax", "cheatsheet"]


def _shape(msa):
    """The (sequences, positions, channels) of an alignment tensor."""
    s = len(msa)
    if s == 0:
        raise ValueError("an alignment with no sequences has nothing to say")
    r = len(msa[0])
    if r == 0:
        raise ValueError("an alignment with no positions has nothing to say")
    c = len(msa[0][0])
    for row in msa:
        if len(row) != r:
            raise ValueError("every sequence must have the same length")
        for v in row:
            if len(v) != c:
                raise ValueError("every position must have the same width")
    return s, r, c


def softmax(logits):
    """Softmax, shifted by the maximum before exponentiating.

    The shift is not a nicety. Attention logits carrying a large pair
    bias overflow the exponential without it, and the shift is exactly
    cancelled by the normalisation, so it costs nothing.
    """
    m = logits[0]
    for v in logits:
        if v > m:
            m = v
    ex = [math.exp(v - m) for v in logits]
    tot = _w.csum(ex)
    return [v / tot for v in ex]


def outer_product_mean(msa):
    """MSA -> pair. AlphaFold 2 Supplementary Algorithm 10.

    Returns an r x r grid of length-(c*c) vectors, the outer product of
    the two positions' channel vectors averaged over the alignment. The
    published block applies a layer normalisation and two learned
    projections to the alignment first and one more to the result; with
    no weights in hand those are the identity, and the caller who has
    them applies them on either side of this call.

    Flattened in row-major order, so entry a*c+b is channel a of
    position i against channel b of position j.
    """
    s, r, c = _shape(msa)
    out = []
    for i in range(r):
        row = []
        for j in range(r):
            cell = []
            for a in range(c):
                for b in range(c):
                    cell.append(_w.csum(msa[k][i][a] * msa[k][j][b]
                                        for k in range(s)) / s)
            row.append(cell)
        out.append(row)
    return out


def pair_bias(pair, w=None):
    """pair -> a scalar per ordered position pair, for the attention.

    In the trained network this is a linear map with no bias term from
    the pair channel down to one scalar per attention head. With no
    weights, the mean over the pair channels stands in for it -- that is
    this module's choice, not AlphaFold's, and it is the linear map with
    every weight equal to 1/c, which keeps the bias on the scale of the
    representation instead of growing with its width.
    """
    r = len(pair)
    out = []
    for i in range(r):
        row = []
        for j in range(r):
            z = pair[i][j]
            if w is None:
                row.append(_w.csum(z) / len(z))
            else:
                row.append(_w.dot(z, w))
        out.append(row)
    return out


def msa_row_attention(msa, bias, scale=None, gate=None):
    """pair -> MSA. AlphaFold 2 Supplementary Algorithm 7, one head.

    Within each sequence of the alignment, every position attends over
    every position of the same sequence, with the pair bias added to the
    logit before the softmax. Query, key and value are the identity
    here for the reason given in the module docstring.

    ``gate``, if given, is a per-channel multiplier applied to the
    attended output, standing in for the learned sigmoid gate; absent,
    the output is ungated.

    Multi-head attention is this operation on a slice of the channel
    axis, so a caller wanting h heads calls it h times on the slices
    and concatenates -- there is no head-specific arithmetic to hide.
    """
    s, r, c = _shape(msa)
    if scale is None:
        scale = 1.0 / math.sqrt(float(c))
    scale = float(scale)
    if len(bias) != r or any(len(row) != r for row in bias):
        raise ValueError("the bias must be one scalar per ordered pair of "
                         "positions")
    if gate is not None and len(gate) != c:
        raise ValueError("the gate must be one multiplier per channel")
    attn = []
    out = []
    for k in range(s):
        seq = msa[k]
        A = []
        O = []
        for i in range(r):
            logits = [_w.dot(seq[i], seq[j]) * scale + bias[i][j]
                      for j in range(r)]
            a = softmax(logits)
            A.append(a)
            row = [_w.csum(a[j] * seq[j][d] for j in range(r))
                   for d in range(c)]
            if gate is not None:
                row = [row[d] * float(gate[d]) for d in range(c)]
            O.append(row)
        attn.append(A)
        out.append(O)
    return attn, out


def openfold_msa_pair(msa, pair, w_bias=None, w_opm=None, scale=None,
                      gate=None):
    """One Evoformer MSA-pair head, both directions.

    Parameters
    ----------
    msa : sequence
        The alignment, indexed [sequence][position][channel].
    pair : sequence
        The pair representation, indexed [i][j][channel].
    w_bias : sequence or None
        The learned map from a pair vector to the attention bias. None
        means the mean over the pair channels; see ``pair_bias``.
    w_opm : sequence of sequences or None
        The learned projection from the c*c outer product back down to
        the pair width, one row per output channel. None means the pair
        representation is returned unchanged and ``pair_updated`` is
        False -- because inventing that projection would be inventing
        the model.
    scale : float or None
        The attention temperature; None is the usual one over root c.
    gate : sequence or None
        A per-channel output gate; None is ungated.

    Returns
    -------
    RichResult
        The outer product mean, the attention bias and weights, the
        updated alignment, and the pair representation.

    References
    ----------
    Jumper et al. (2021) Nature 596, 583-589, Supplementary Algorithms
    7 and 10; Ahdritz et al. (2022) bioRxiv 2022.11.20.517210.
    """
    s, r, c = _shape(msa)
    if len(pair) != r or any(len(row) != r for row in pair):
        raise ValueError("the pair representation must be square over the "
                         "alignment's positions")
    cz = len(pair[0][0])

    opm = outer_product_mean(msa)
    b = pair_bias(pair, w_bias)
    attn, msa_out = msa_row_attention(msa, b, scale, gate)

    if w_opm is None:
        pair_out = [[list(pair[i][j]) for j in range(r)] for i in range(r)]
        updated = False
    else:
        if any(len(row) != c * c for row in w_opm):
            raise ValueError("each projection row must span the whole outer "
                             "product")
        if len(w_opm) != cz:
            raise ValueError("the projection must land on the pair width")
        pair_out = [[[pair[i][j][d] + _w.dot(w_opm[d], opm[i][j])
                      for d in range(cz)] for j in range(r)]
                    for i in range(r)]
        updated = True

    # The mass the alignment puts on the diagonal: how much each
    # position attends to itself rather than to its neighbours. It is
    # the one summary of the attention that reads the same regardless of
    # how many sequences or positions there are.
    diag = _w.csum(attn[k][i][i] for k in range(s) for i in range(r))
    return RichResult(payload={
        "opm": opm,
        "bias": b,
        "attn": attn,
        "msa_out": msa_out,
        "pair_out": pair_out,
        "pair_updated": updated,
        "self_attention": diag / float(s * r),
        "n_seq": s,
        "n_pos": r,
        "n_channel": c,
        "n_pair_channel": cz,
        "scale": (1.0 / math.sqrt(float(c))) if scale is None
                 else float(scale),
        "gated": gate is not None,
        "method": "OpenFold Evoformer MSA-pair head",
    })


def cheatsheet():
    return ("alfomg: OpenFold MSA-pair head. Outer product mean for "
            "MSA->pair, row attention with a pair bias for pair->MSA; "
            "trained weights are parameters, not constants")
