r"""Variant effect annotation against a coding sequence.

Cingolani, P., et al. (2012) "A program for annotating and predicting the
effects of single nucleotide polymorphisms, SnpEff", *Fly* 6(2), 80-92.

A variant is a change to a genome; what a biologist needs is what the
change *does*. The classification is decided by where the variant falls
in a transcript and, for coding variants, by what the codon becomes:

============================  ==================================
``synonymous_variant``        the codon changes, the amino acid does not
``missense_variant``          a different amino acid
``stop_gained``               a codon becomes a stop (the paper's HIGH
                              impact class)
``stop_lost``                 the stop codon becomes an amino acid
``start_lost``                the initiator ATG is destroyed
``frameshift_variant``        an indel whose length is not a multiple of 3
``inframe_insertion`` /       an indel that keeps the reading frame
``inframe_deletion``
``intron_variant``,           outside the coding sequence
``upstream_gene_variant``,
``downstream_gene_variant``,
``intergenic_variant``
============================  ==================================

The impact grades are the paper's own four: HIGH, MODERATE, LOW and
MODIFIER. Translation uses the standard genetic code (NCBI table 1),
which is written out in full below rather than derived, because it is a
lookup table and any cleverness would only be a way to get it wrong.
"""

from ._richresult import RichResult

__all__ = ["snpeff", "translate", "codon_table", "annotate_variant"]

_BASES = "TCAG"
_AA = ("FFLLSSSSYY**CC*W"
       "LLLLPPPPHHQQRRRR"
       "IIIMTTTTNNKKSSRR"
       "VVVVAAAADDEEGGGG")

_CODONS = {}
for _i, _b1 in enumerate(_BASES):
    for _j, _b2 in enumerate(_BASES):
        for _k, _b3 in enumerate(_BASES):
            _CODONS[_b1 + _b2 + _b3] = _AA[_i * 16 + _j * 4 + _k]

_HIGH = ("stop_gained", "stop_lost", "start_lost", "frameshift_variant")
_MODERATE = ("missense_variant", "inframe_insertion",
             "inframe_deletion")
_LOW = ("synonymous_variant", "stop_retained_variant")


def codon_table():
    """The standard genetic code, codon -> one-letter amino acid."""
    return dict(_CODONS)


def translate(seq, to_stop=False):
    """Translate a nucleotide sequence in frame 0."""
    s = str(seq).upper().replace("U", "T")
    for ch in s:
        if ch not in "ACGTN":
            raise ValueError("snpeff: %r is not a nucleotide" % ch)
    out = []
    for i in range(0, len(s) - 2, 3):
        aa = _CODONS.get(s[i:i + 3], "X")
        if to_stop and aa == "*":
            break
        out.append(aa)
    return "".join(out)


def _impact(effect):
    if effect in _HIGH:
        return "HIGH"
    if effect in _MODERATE:
        return "MODERATE"
    if effect in _LOW:
        return "LOW"
    return "MODIFIER"


def annotate_variant(cds, pos, ref, alt, cds_start=0, upstream=5000,
                     downstream=5000, transcript_len=None):
    """Classify one variant against a coding sequence.

    ``pos`` is 0-based in the sequence the caller passes; ``cds_start``
    says where the coding sequence begins within it, so a variant before
    that is upstream and one past the stop is downstream.
    """
    seq = str(cds).upper().replace("U", "T")
    ref = str(ref).upper()
    alt = str(alt).upper()
    pos = int(pos)
    if not seq:
        raise ValueError("snpeff: the sequence is empty")
    if not ref or not alt:
        raise ValueError("snpeff: ref and alt must be non-empty")
    if pos < 0 or pos >= len(seq):
        raise ValueError("snpeff: position %d is outside the sequence"
                         % pos)
    if seq[pos:pos + len(ref)] != ref:
        raise ValueError("snpeff: the reference allele %r does not match "
                         "the sequence at position %d (%r)"
                         % (ref, pos, seq[pos:pos + len(ref)]))
    end = len(seq) if transcript_len is None else cds_start + \
        transcript_len

    if pos < cds_start:
        d = cds_start - pos
        eff = ("upstream_gene_variant" if d <= upstream
               else "intergenic_variant")
        return _pack(eff, None, None, None, None, ref, alt, pos)
    if pos >= end:
        d = pos - end + 1
        eff = ("downstream_gene_variant" if d <= downstream
               else "intergenic_variant")
        return _pack(eff, None, None, None, None, ref, alt, pos)

    coding = seq[cds_start:end]
    off = pos - cds_start
    mutated = coding[:off] + alt + coding[off + len(ref):]

    if len(ref) != len(alt):
        shift = (len(alt) - len(ref)) % 3
        if shift:
            eff = "frameshift_variant"
        else:
            eff = ("inframe_insertion" if len(alt) > len(ref)
                   else "inframe_deletion")
        return _pack(eff, None, None, translate(coding),
                     translate(mutated), ref, alt, pos,
                     codon_index=off // 3)

    ci = off // 3
    ref_codon = coding[ci * 3:ci * 3 + 3]
    alt_codon = mutated[ci * 3:ci * 3 + 3]
    if len(ref_codon) < 3 or len(alt_codon) < 3:
        raise ValueError("snpeff: the coding sequence is not a whole "
                         "number of codons at position %d" % pos)
    ra, aa = _CODONS.get(ref_codon, "X"), _CODONS.get(alt_codon, "X")
    if ci == 0 and ra == "M" and aa != "M":
        eff = "start_lost"
    elif ra == "*" and aa != "*":
        eff = "stop_lost"
    elif ra != "*" and aa == "*":
        eff = "stop_gained"
    elif ra == aa:
        eff = ("stop_retained_variant" if ra == "*"
               else "synonymous_variant")
    else:
        eff = "missense_variant"
    return _pack(eff, ref_codon, alt_codon, ra, aa, ref, alt, pos,
                 codon_index=ci,
                 hgvs_p="p.%s%d%s" % (ra, ci + 1, aa) if ra != aa
                 else "p.%s%d=" % (ra, ci + 1))


def _pack(effect, ref_codon, alt_codon, ref_aa, alt_aa, ref, alt, pos,
          codon_index=None, hgvs_p=None):
    return {
        "effect": effect,
        "impact": _impact(effect),
        "ref_codon": ref_codon,
        "alt_codon": alt_codon,
        "ref_aa": ref_aa,
        "alt_aa": alt_aa,
        "codon_index": codon_index,
        "hgvs_p": hgvs_p,
        "hgvs_c": "c.%d%s>%s" % (pos + 1, ref, alt),
        "pos": pos,
        "ref": ref,
        "alt": alt,
    }


def snpeff(cds, variants, cds_start=0, upstream=5000, downstream=5000,
           transcript_len=None):
    """Annotate a list of ``(pos, ref, alt)`` variants."""
    out = []
    for v in variants:
        if len(v) != 3:
            raise ValueError("snpeff: each variant must be "
                             "(pos, ref, alt)")
        out.append(annotate_variant(cds, v[0], v[1], v[2], cds_start,
                                    upstream, downstream,
                                    transcript_len))
    counts = {}
    for a in out:
        counts[a["effect"]] = counts.get(a["effect"], 0) + 1
    impacts = {}
    for a in out:
        impacts[a["impact"]] = impacts.get(a["impact"], 0) + 1
    return RichResult(payload={
        "estimate": out,
        "annotations": out,
        "effect_counts": counts,
        "impact_counts": impacts,
        "n_variants": len(out),
        "protein": translate(str(cds).upper()[cds_start:]
                             if transcript_len is None else
                             str(cds).upper()[cds_start:cds_start +
                                              transcript_len]),
        "method": ("variant effect annotation (Cingolani et al. 2012, "
                   "SnpEff), standard genetic code"),
        "note": ("impact grades are the paper's HIGH / MODERATE / LOW / "
                 "MODIFIER; positions are 0-based and hgvs_c is "
                 "1-based, as the notation requires"),
    })


def cheatsheet():
    return ("snpeff: variant annotation (Cingolani et al. 2012). "
            "Classify by codon change: synonymous, missense, "
            "stop_gained, stop_lost, start_lost; by indel length mod 3: "
            "frameshift against inframe; by position: upstream, "
            "downstream, intergenic. Impact HIGH for the four that "
            "break the protein, MODERATE for missense and inframe "
            "indels, LOW for synonymous, MODIFIER for the rest.")
