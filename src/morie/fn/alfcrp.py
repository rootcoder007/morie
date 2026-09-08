# morie.fn -- function file (rootcoder007/morie)
"""Contiguous residue cropping of AlphaFold training features."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["alphafold_cropping"]


def alphafold_cropping(seqlen, cropsize, start=1, target=None, pair=None,
                       msa=None, mode="clamped"):
    """Residue cropping -- supplement section 1.2.8, pp. 7-8.

    Training crops the residue dimension of every feature to one contiguous
    window.  Because the window is contiguous and shared by all features,
    the pair features must be cropped on both axes with the same index set,
    which is what keeps residue ``i`` of the cropped target aligned with row
    ``i`` of the cropped pair tensor.

    The crop start is an argument, not sampled here, so the function is
    deterministic.  The valid range the spec would sample from is returned
    alongside the crop: in clamped mode the start is uniform on
    ``[1, n + 1]``, and in unclamped mode on ``[1, n - x + 1]`` where ``n``
    is ``seqlen - cropsize`` and ``x`` is itself uniform on ``[0, n]``.

    Parameters
    ----------
    seqlen : int
        Number of residues before cropping.
    cropsize : int
        Number of residues to keep.
    start : int
        One-based index of the first residue kept.
    target : list, optional
        Per-residue features to crop along their first axis.
    pair : list of list, optional
        Pair features to crop along both axes.
    msa : list of list, optional
        MSA features, cropped along their residue axis (the second).
    mode : {"clamped", "unclamped"}
        Which sampling rule to report the valid start range for.

    Returns
    -------
    result : RichResult
        Keys: ``idx`` (the zero-based indices kept), ``target``, ``pair``,
        ``msa`` (cropped, or ``None``), ``startmax``, ``estimate``
        (``cropsize``), ``method``.

    Notes
    -----
    Cropping the full length from position one is the identity on every
    feature, and the cropped pair tensor is always exactly the submatrix of
    the original on the kept indices.  The harness checks both.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary section 1.2.8
    """
    if mode not in ("clamped", "unclamped"):
        raise ValueError("mode must be 'clamped' or 'unclamped'")
    if cropsize > seqlen:
        raise ValueError("cropsize %d exceeds seqlen %d" % (cropsize, seqlen))
    if start < 1 or start + cropsize - 1 > seqlen:
        raise ValueError("crop [%d, %d] falls outside 1..%d"
                         % (start, start + cropsize - 1, seqlen))

    nn = seqlen - cropsize
    # clamped: Uniform[1, n + 1]; unclamped: Uniform[1, n - x + 1], whose
    # widest case is x = 0 and so coincides with the clamped upper limit
    startmax = nn + 1
    idx = list(range(start - 1, start - 1 + cropsize))

    ct = None if target is None else [target[i] for i in idx]
    cp = None if pair is None else [[pair[i][j] for j in idx] for i in idx]
    cmsa = None if msa is None else [[row[i] for i in idx] for row in msa]

    return RichResult(
        payload={
            "idx": idx,
            "target": ct,
            "pair": cp,
            "msa": cmsa,
            "startmax": startmax,
            "estimate": float(cropsize),
            "mode": mode,
            "method": "AlphaFold contiguous residue cropping",
        }
    )


def cheatsheet():
    return "alfcrp: contiguous residue crop shared across all features"
