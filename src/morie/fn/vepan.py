# morie.fn -- function file (rootcoder007/morie)
r"""The Ensembl Variant Effect Predictor: consequences of a variant.

McLaren, W., Gil, L., Hunt, S. E., Riat, H. S., Ritchie, G. R. S.,
Thormann, A., Flicek, P., & Cunningham, F. (2016) "The Ensembl Variant
Effect Predictor", *Genome Biology* 17:122. doi:10.1186/s13059-016-0974-4

The VEP answers one question per (variant, transcript) pair: what does
this allele do to this transcript? The paper's own account of the
machinery (its "Implementation" section) is that for each overlap of a
variant with a feature the API builds an object per allele and then
"evaluates consequence types using a set of predicate functions" -- a
missense predicate, a splice-donor predicate, and so on -- so that one
allele can carry several terms at once, "such as a variant that falls in
a splice-relevant region that also affects the coding sequence".

**The terms are not free-form.** Section "Variant consequences" says they
come from "a standardized set of variant annotation terms [67] which
were defined in collaboration with the Sequence Ontology", each with a
stable identifier. Reference [67] is Ensembl's published consequence
table, and the severity ordering used here is that table's -- 41 terms,
rank 1 (``transcript_ablation``) to rank 41 (``sequence_variant``), with
the HIGH / MODERATE / LOW / MODIFIER impact of each. It is reproduced in
:data:`CONSEQUENCE_RANK` with the SO accessions, because a severity order
invented locally would make ``most_severe_consequence`` and ``--pick``
mean nothing.

**Picking one line per variant.** Table 7 defines ``--pick`` as: "priority
is given to the canonical transcript for each gene, protein coding
transcripts, and more severe consequence types". That is an ordered list,
not a tie-break on severity -- the canonical transcript wins even when a
different transcript carries a worse consequence. ``--per_gene`` is the
same rule applied within each gene. Both are here as
``pick(..., per_gene=False|True)``, and the ordering can be inspected
through :data:`PICK_ORDER`.

Which consequences are computed
-------------------------------
Everything a transcript model plus a reference sequence can decide:
``splice_acceptor_variant``, ``splice_donor_variant``, ``stop_gained``,
``frameshift_variant``, ``stop_lost``, ``start_lost``,
``inframe_insertion``, ``inframe_deletion``, ``missense_variant``,
``splice_donor_5th_base_variant``, ``splice_region_variant``,
``splice_polypyrimidine_tract_variant``, ``start_retained_variant``,
``stop_retained_variant``, ``synonymous_variant``,
``coding_sequence_variant``, ``5_prime_UTR_variant``,
``3_prime_UTR_variant``, ``non_coding_transcript_exon_variant``,
``intron_variant``, ``non_coding_transcript_variant``,
``upstream_gene_variant``, ``downstream_gene_variant`` and
``intergenic_variant``.

Not computed, because they need data a sequence and a gene model do not
carry: the regulatory and TF-binding terms (rank 34-39, they need a
regulatory build), ``NMD_transcript_variant`` (needs the transcript's NMD
flag), ``transcript_ablation`` / ``transcript_amplification`` (structural
variants), and ``mature_miRNA_variant``. They are in the rank table so
that severity comparisons against externally supplied terms still work.

HGVS
----
``c.`` and ``p.`` notations are produced from the transcript
coordinates. The paper notes these "undergo significant additional
processing to conform to the nomenclature definition", in particular
that an indel in repetitive sequence "must be reported at the most 3'
position possible"; that 3' shift is applied here.

Coordinates are VCF-style: ``pos`` is 1-based and ``ref``/``alt`` share a
leading anchor base for indels.
"""

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult
from .snpeff import translate

__all__ = [
    "vep_annotation",
    "vepannotation",
    "annotate",
    "pick",
    "most_severe_consequence",
    "consequence_rank",
    "consequence_impact",
    "transcript_sequence",
    "CONSEQUENCE_RANK",
    "CONSEQUENCE_IMPACT",
    "CONSEQUENCE_SO",
    "PICK_ORDER",
]

# Ensembl's published consequence table (reference [67] of the paper):
# (rank, SO term, impact, SO accession). The rank IS the severity order
# that most_severe_consequence and --pick depend on.
_TABLE = (
    (1, "transcript_ablation", "HIGH", "SO:0001893"),
    (2, "splice_acceptor_variant", "HIGH", "SO:0001574"),
    (3, "splice_donor_variant", "HIGH", "SO:0001575"),
    (4, "stop_gained", "HIGH", "SO:0001587"),
    (5, "frameshift_variant", "HIGH", "SO:0001589"),
    (6, "stop_lost", "HIGH", "SO:0001578"),
    (7, "start_lost", "HIGH", "SO:0002012"),
    (8, "transcript_amplification", "HIGH", "SO:0001889"),
    (9, "feature_elongation", "HIGH", "SO:0001907"),
    (10, "feature_truncation", "HIGH", "SO:0001906"),
    (11, "inframe_insertion", "MODERATE", "SO:0001821"),
    (12, "inframe_deletion", "MODERATE", "SO:0001822"),
    (13, "missense_variant", "MODERATE", "SO:0001583"),
    (14, "protein_altering_variant", "MODERATE", "SO:0001818"),
    (15, "splice_donor_5th_base_variant", "LOW", "SO:0001787"),
    (16, "splice_region_variant", "LOW", "SO:0001630"),
    (17, "splice_donor_region_variant", "LOW", "SO:0002170"),
    (18, "splice_polypyrimidine_tract_variant", "LOW", "SO:0002169"),
    (19, "incomplete_terminal_codon_variant", "LOW", "SO:0001626"),
    (20, "start_retained_variant", "LOW", "SO:0002019"),
    (21, "stop_retained_variant", "LOW", "SO:0001567"),
    (22, "synonymous_variant", "LOW", "SO:0001819"),
    (23, "coding_sequence_variant", "MODIFIER", "SO:0001580"),
    (24, "mature_miRNA_variant", "MODIFIER", "SO:0001620"),
    (25, "5_prime_UTR_variant", "MODIFIER", "SO:0001623"),
    (26, "3_prime_UTR_variant", "MODIFIER", "SO:0001624"),
    (27, "non_coding_transcript_exon_variant", "MODIFIER", "SO:0001792"),
    (28, "intron_variant", "MODIFIER", "SO:0001627"),
    (29, "NMD_transcript_variant", "MODIFIER", "SO:0001621"),
    (30, "non_coding_transcript_variant", "MODIFIER", "SO:0001619"),
    (31, "coding_transcript_variant", "MODIFIER", "SO:0001968"),
    (32, "upstream_gene_variant", "MODIFIER", "SO:0001631"),
    (33, "downstream_gene_variant", "MODIFIER", "SO:0001632"),
    (34, "TFBS_ablation", "MODERATE", "SO:0001895"),
    (35, "TFBS_amplification", "MODIFIER", "SO:0001892"),
    (36, "TF_binding_site_variant", "MODIFIER", "SO:0001782"),
    (37, "regulatory_region_ablation", "MODIFIER", "SO:0001894"),
    (38, "regulatory_region_amplification", "MODIFIER", "SO:0001891"),
    (39, "regulatory_region_variant", "MODIFIER", "SO:0001566"),
    (40, "intergenic_variant", "MODIFIER", "SO:0001628"),
    (41, "sequence_variant", "MODIFIER", "SO:0001060"),
)

CONSEQUENCE_RANK = dict((t, r) for r, t, _, _ in _TABLE)
CONSEQUENCE_IMPACT = dict((t, i) for _, t, i, _ in _TABLE)
CONSEQUENCE_SO = dict((t, a) for _, t, _, a in _TABLE)

#: Table 7's ``--pick`` order, applied in sequence. Canonical first --
#: a canonical transcript wins even against a worse consequence
#: elsewhere.
PICK_ORDER = ("canonical", "protein_coding", "consequence_rank",
              "transcript_id")

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
_AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
        "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
        "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
        "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
        "*": "Ter", "X": "Xaa"}


def consequence_rank(term):
    """Severity rank of an SO term; 1 is worst. Unknown terms sort last."""
    return CONSEQUENCE_RANK.get(term, len(_TABLE) + 1)


def consequence_impact(term):
    """``HIGH``/``MODERATE``/``LOW``/``MODIFIER`` for an SO term."""
    return CONSEQUENCE_IMPACT.get(term, "MODIFIER")


def most_severe_consequence(terms):
    """The lowest-ranked (worst) term in ``terms``."""
    ts = [str(t) for t in terms]
    if not ts:
        raise ValueError("vepan: no consequence terms given")
    return min(ts, key=lambda t: (consequence_rank(t), t))


def _revcomp(s):
    return "".join(_COMPLEMENT.get(c, "N") for c in reversed(s.upper()))


# ------------------------------------------------------ transcript model

def _transcript(tr):
    try:
        exons = [(int(a), int(b)) for a, b in tr["exons"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("vepan: a transcript needs exons as (start, end)")
    if not exons:
        raise ValueError("vepan: a transcript needs at least one exon")
    exons.sort()
    for a, b in exons:
        if a > b or a < 1:
            raise ValueError("vepan: exon coordinates must be 1-based and "
                             "ascending")
    for k in range(len(exons) - 1):
        if exons[k][1] >= exons[k + 1][0]:
            raise ValueError("vepan: exons must not overlap")
    strand = tr.get("strand", "+")
    if strand not in ("+", "-"):
        raise ValueError("vepan: strand must be '+' or '-'")
    cs, ce = tr.get("cds_start"), tr.get("cds_end")
    biotype = tr.get("biotype", "protein_coding" if cs else "lncRNA")
    if biotype == "protein_coding" and (cs is None or ce is None):
        raise ValueError("vepan: a protein_coding transcript needs "
                         "cds_start and cds_end")
    if cs is not None and ce is not None and int(cs) > int(ce):
        raise ValueError("vepan: cds_start must not exceed cds_end")
    return {"id": tr.get("id", "TR"), "gene": tr.get("gene", "GENE"),
            "chrom": tr.get("chrom", "chr1"), "strand": strand,
            "exons": exons, "cds_start": None if cs is None else int(cs),
            "cds_end": None if ce is None else int(ce), "biotype": biotype,
            "canonical": bool(tr.get("canonical", False)),
            "start": exons[0][0], "end": exons[-1][1]}


def transcript_sequence(tr, genome):
    """The spliced transcript, 5' to 3', and the genomic position of each
    of its bases."""
    t = _transcript(tr)
    seq, gpos = [], []
    for a, b in t["exons"]:
        seq.append(str(genome)[a - 1:b])
        gpos.extend(range(a, b + 1))
    s = "".join(seq).upper()
    if t["strand"] == "-":
        s = _revcomp(s)
        gpos = list(reversed(gpos))
    return s, gpos


def _cds_frame(t, genome):
    """cDNA index of each coding base, and the coding sequence itself."""
    if t["cds_start"] is None:
        return None, None, None
    seq, gpos = transcript_sequence(t, genome)
    coding = [k for k, g in enumerate(gpos)
              if t["cds_start"] <= g <= t["cds_end"]]
    if not coding:
        raise ValueError("vepan: the CDS does not overlap any exon")
    cds = "".join(seq[k] for k in coding)
    return cds, coding, (seq, gpos)


def _introns(t):
    return [(t["exons"][k][1] + 1, t["exons"][k + 1][0] - 1)
            for k in range(len(t["exons"]) - 1)
            if t["exons"][k + 1][0] - t["exons"][k][1] > 1]


def _variant(v):
    try:
        pos = int(v["pos"])
        ref = str(v["ref"]).upper()
        alt = str(v["alt"]).upper()
    except (KeyError, TypeError, ValueError):
        raise ValueError("vepan: a variant needs pos, ref and alt")
    if pos < 1:
        raise ValueError("vepan: pos is 1-based and must be positive")
    if not ref or not alt:
        raise ValueError("vepan: ref and alt must be non-empty")
    for base in ref + alt:
        if base not in "ACGTN":
            raise ValueError("vepan: ref and alt must be ACGTN")
    if len(ref) > 1 and len(alt) > 1:
        raise ValueError("vepan: only SNVs and anchored indels are handled")
    return {"chrom": v.get("chrom", "chr1"), "pos": pos, "ref": ref,
            "alt": alt, "id": v.get("id")}


def _kind(v):
    if len(v["ref"]) == len(v["alt"]) == 1:
        return "SNV"
    return "insertion" if len(v["alt"]) > len(v["ref"]) else "deletion"


def _affected(v):
    """Genomic bases the variant changes (1-based, inclusive)."""
    if _kind(v) == "SNV":
        return v["pos"], v["pos"]
    if _kind(v) == "insertion":
        # the inserted bases sit between pos and pos+1
        return v["pos"] + 1, v["pos"]
    return v["pos"] + 1, v["pos"] + len(v["ref"]) - 1


# ------------------------------------------------------- the predicates

def _splice_terms(t, v, lo, hi):
    """Splice consequences from the distance to each exon boundary.

    The Sequence Ontology definitions, in transcript orientation: the
    donor is the first two intronic bases after an exon and the acceptor
    the last two before one; ``splice_region_variant`` reaches 1-3 bases
    into the exon and 3-8 into the intron; the polypyrimidine tract is
    intronic positions -17 to -3 before an acceptor; and the donor 5th
    base has its own term.
    """
    terms = set()
    fwd = t["strand"] == "+"
    for k, (ia, ib) in enumerate(_introns(t)):
        # donor side of this intron in transcript orientation
        d_start, a_end = (ia, ib) if fwd else (ib, ia)
        step = 1 if fwd else -1
        for p in range(lo, hi + 1):
            if not (min(ia, ib) <= p <= max(ia, ib)):
                continue
            d = (p - d_start) * step + 1        # 1-based into the intron
            a = (a_end - p) * step + 1          # 1-based back from the end
            if d <= 2:
                terms.add("splice_donor_variant")
            elif d == 5:
                terms.add("splice_donor_5th_base_variant")
            if a <= 2:
                terms.add("splice_acceptor_variant")
            if 3 <= d <= 8:
                terms.add("splice_region_variant")
            if 3 <= a <= 8:
                terms.add("splice_region_variant")
            if 3 <= a <= 17:
                terms.add("splice_polypyrimidine_tract_variant")
    for a, b in t["exons"]:
        for p in range(lo, hi + 1):
            if a <= p <= b:
                near = min(p - a + 1, b - p + 1)
                # only next to a real intron, not at the transcript ends
                touches = ((a != t["start"] and p - a + 1 <= 3) or
                           (b != t["end"] and b - p + 1 <= 3))
                if near <= 3 and touches:
                    terms.add("splice_region_variant")
    return terms


def _apply(cds, idx, v, coding, gpos, strand):
    """The coding sequence after the variant, and where it changed."""
    # map the genomic bases of the variant onto CDS offsets
    pos_map = dict((gpos[k], off) for off, k in enumerate(coding))
    lo, hi = _affected(v)
    if _kind(v) == "insertion":
        anchor = pos_map.get(v["pos"])
        if anchor is None:
            return None, None
        ins = v["alt"][1:]
        if strand == "-":
            ins = _revcomp(ins)
            cut = anchor            # inserted before the anchor in CDS order
        else:
            cut = anchor + 1
        return cds[:cut] + ins + cds[cut:], cut
    offs = sorted(pos_map[p] for p in range(lo, hi + 1) if p in pos_map)
    if not offs:
        return None, None
    if _kind(v) == "deletion":
        keep = [c for k, c in enumerate(cds) if k not in set(offs)]
        return "".join(keep), offs[0]
    base = v["alt"] if strand == "+" else _COMPLEMENT[v["alt"]]
    o = offs[0]
    return cds[:o] + base + cds[o + 1:], o


def _coding_terms(t, v, cds, coding, gpos):
    """Everything that depends on the protein: the predicate set."""
    terms = set()
    alt_cds, off = _apply(cds, None, v, coding, gpos, t["strand"])
    if alt_cds is None:
        return terms, {}
    ref_prot = translate(cds)
    alt_prot = translate(alt_cds)
    kind = _kind(v)
    delta = len(alt_cds) - len(cds)
    codon = off // 3
    info = {"cds_position": off + 1, "protein_position": codon + 1,
            "ref_codon": cds[codon * 3:codon * 3 + 3],
            "ref_aa": ref_prot[codon] if codon < len(ref_prot) else "",
            "alt_aa": alt_prot[codon] if codon < len(alt_prot) else "",
            "cds_length": len(cds), "alt_cds_length": len(alt_cds)}
    if kind != "SNV" and delta % 3 != 0:
        terms.add("frameshift_variant")
        if "*" in alt_prot[:codon + 1]:
            terms.add("stop_gained")
        return terms, info
    if kind == "insertion":
        terms.add("inframe_insertion")
    elif kind == "deletion":
        terms.add("inframe_deletion")

    ref_stop = ref_prot.find("*")
    alt_stop = alt_prot.find("*")
    if codon == 0:
        if info["alt_aa"] == "M" and info["ref_aa"] == "M":
            terms.add("start_retained_variant")
        elif info["ref_aa"] == "M":
            terms.add("start_lost")
    if ref_stop >= 0 and codon == ref_stop:
        if info["alt_aa"] == "*":
            terms.add("stop_retained_variant")
        else:
            terms.add("stop_lost")
    # An in-frame indel moves the existing stop by delta/3 codons without
    # gaining one; only a stop EARLIER than that counts as stop_gained.
    expected_stop = ref_stop + delta // 3 if ref_stop >= 0 else -1
    if (alt_stop >= 0 and (expected_stop < 0 or alt_stop < expected_stop) and
            "stop_lost" not in terms):
        terms.add("stop_gained")
    if kind == "SNV" and not terms:
        if info["ref_aa"] == info["alt_aa"]:
            terms.add("synonymous_variant")
        else:
            terms.add("missense_variant")
    if not terms:
        terms.add("coding_sequence_variant")
    return terms, info


def _hgvs_c(t, v, seq, gpos, cds_first):
    """``c.`` notation, with an indel shifted to its most 3' position."""
    idx = {}
    for k, g in enumerate(gpos):
        idx[g] = k
    lo, hi = _affected(v)
    anchor = idx.get(v["pos"])
    if anchor is None:
        return None
    kind = _kind(v)

    def c_of(k):
        return k - cds_first + 1 if cds_first is not None else k + 1

    if kind == "SNV":
        k = idx[v["pos"]]
        ref = seq[k]
        alt = v["alt"] if t["strand"] == "+" else _COMPLEMENT[v["alt"]]
        return "c.%d%s>%s" % (c_of(k), ref, alt)
    if kind == "insertion":
        ins = v["alt"][1:]
        if t["strand"] == "-":
            ins = _revcomp(ins)
        k = anchor if t["strand"] == "-" else anchor + 1
        # 3' shift: slide the insertion right while it repeats
        while k + len(ins) <= len(seq) and seq[k:k + len(ins)] == ins:
            k += len(ins)
        return "c.%d_%dins%s" % (c_of(k - 1), c_of(k), ins)
    ks = sorted(idx[p] for p in range(lo, hi + 1) if p in idx)
    if not ks:
        return None
    a, b = ks[0], ks[-1]
    n = b - a + 1
    while b + n < len(seq) and seq[a:b + 1] == seq[a + n:b + 1 + n]:
        a, b = a + n, b + n
    dele = seq[a:b + 1]
    if a == b:
        return "c.%ddel%s" % (c_of(a), dele)
    return "c.%d_%ddel%s" % (c_of(a), c_of(b), dele)


def _hgvs_p(info, terms):
    if not info or "protein_position" not in info:
        return None
    ref, alt = info.get("ref_aa"), info.get("alt_aa")
    pos = info["protein_position"]
    if "frameshift_variant" in terms:
        return "p.%s%dfs" % (_AA3.get(ref, "Xaa"), pos)
    if not ref:
        return None
    if "synonymous_variant" in terms or "stop_retained_variant" in terms \
            or "start_retained_variant" in terms:
        return "p.%s%d=" % (_AA3.get(ref, "Xaa"), pos)
    if not alt:
        return None
    return "p.%s%d%s" % (_AA3.get(ref, "Xaa"), pos, _AA3.get(alt, "Xaa"))


# ------------------------------------------------------------- annotate

def annotate(variant, transcripts, genome, upstream=5000, downstream=5000):
    """One record per (variant, transcript) overlap, as the VEP emits.

    A variant beyond every transcript's flank gets a single
    ``intergenic_variant`` record with no transcript attached.
    """
    v = _variant(variant)
    trs = [_transcript(t) for t in transcripts]
    if upstream < 0 or downstream < 0:
        raise ValueError("vepan: flank sizes must be non-negative")
    g = str(genome).upper()
    lo, hi = _affected(v)
    span_lo, span_hi = min(lo, v["pos"]), max(hi, v["pos"])
    out = []
    for t in trs:
        if t["chrom"] != v["chrom"]:
            continue
        five, three = ((upstream, downstream) if t["strand"] == "+"
                       else (downstream, upstream))
        if span_hi < t["start"] - five or span_lo > t["end"] + three:
            continue
        terms = set()
        if span_hi < t["start"]:
            terms.add("upstream_gene_variant" if t["strand"] == "+"
                      else "downstream_gene_variant")
        elif span_lo > t["end"]:
            terms.add("downstream_gene_variant" if t["strand"] == "+"
                      else "upstream_gene_variant")
        info, hgvs_c, hgvs_p = {}, None, None
        if not terms:
            seq, gpos = transcript_sequence(t, g)
            in_exon = any(a <= p <= b for a, b in t["exons"]
                          for p in range(lo, hi + 1)) or \
                (_kind(v) == "insertion" and
                 any(a <= v["pos"] <= b for a, b in t["exons"]))
            in_intron = any(a <= p <= b for a, b in _introns(t)
                            for p in range(lo, hi + 1))
            terms |= _splice_terms(t, v, min(lo, v["pos"]),
                                   max(hi, v["pos"]))
            if in_intron:
                terms.add("intron_variant")
            if t["biotype"] != "protein_coding":
                if in_exon:
                    terms.add("non_coding_transcript_exon_variant")
                terms.add("non_coding_transcript_variant")
            elif in_exon:
                cds, coding, _ = _cds_frame(t, g)
                cds_first = coding[0]
                if all(t["cds_start"] <= p <= t["cds_end"]
                       for p in range(lo, hi + 1)) or \
                        (_kind(v) == "insertion" and
                         t["cds_start"] <= v["pos"] <= t["cds_end"]):
                    ct, info = _coding_terms(t, v, cds, coding, gpos)
                    terms |= ct
                else:
                    utr5 = (v["pos"] < t["cds_start"]) == (t["strand"] == "+")
                    terms.add("5_prime_UTR_variant" if utr5
                              else "3_prime_UTR_variant")
                hgvs_c = _hgvs_c(t, v, seq, gpos, cds_first)
                hgvs_p = _hgvs_p(info, terms)
            if not terms:
                terms.add("intron_variant")
        record = {
            "variant": v.get("id") or "%s:%d%s>%s" % (v["chrom"], v["pos"],
                                                      v["ref"], v["alt"]),
            "transcript": t["id"], "gene": t["gene"],
            "biotype": t["biotype"], "canonical": t["canonical"],
            "strand": t["strand"],
            "consequences": sorted(terms, key=consequence_rank),
            "most_severe": most_severe_consequence(terms),
            "impact": consequence_impact(most_severe_consequence(terms)),
            "hgvs_c": hgvs_c, "hgvs_p": hgvs_p,
        }
        record.update(info)
        out.append(record)
    if not out:
        out.append({
            "variant": v.get("id") or "%s:%d%s>%s" % (v["chrom"], v["pos"],
                                                      v["ref"], v["alt"]),
            "transcript": None, "gene": None, "biotype": None,
            "canonical": False, "strand": None,
            "consequences": ["intergenic_variant"],
            "most_severe": "intergenic_variant", "impact": "MODIFIER",
            "hgvs_c": None, "hgvs_p": None})
    out.sort(key=lambda r: (consequence_rank(r["most_severe"]),
                            r["transcript"] or ""))
    return out


def _pick_key(r):
    """Table 7's order: canonical, then protein coding, then severity."""
    return (0 if r["canonical"] else 1,
            0 if r["biotype"] == "protein_coding" else 1,
            consequence_rank(r["most_severe"]),
            r["transcript"] or "")


def pick(records, per_gene=False):
    """``--pick`` (one record) or ``--per_gene`` (one per gene)."""
    rs = list(records)
    if not rs:
        raise ValueError("vepan: nothing to pick from")
    if not per_gene:
        return [min(rs, key=_pick_key)]
    best = {}
    for r in rs:
        key = r["gene"]
        if key not in best or _pick_key(r) < _pick_key(best[key]):
            best[key] = r
    return [best[k] for k in sorted(best, key=lambda k: (k is None, k))]


def vep_annotation(variants, transcripts, genome, upstream=5000,
                   downstream=5000, mode="all", no_intergenic=False):
    """Annotate every variant against every transcript.

    ``mode`` is ``"all"`` (every overlap, the default), ``"pick"``
    (Table 7's one line per variant) or ``"per_gene"``.
    """
    if mode not in ("all", "pick", "per_gene"):
        raise ValueError("vepan: mode must be 'all', 'pick' or 'per_gene'")
    vs = [_variant(v) for v in variants]
    if not vs:
        raise ValueError("vepan: no variants given")
    trs = list(transcripts)
    rows = []
    for v in vs:
        recs = annotate(v, trs, genome, upstream, downstream)
        if no_intergenic:
            recs = [r for r in recs if r["transcript"] is not None]
            if not recs:
                continue
        if mode == "pick":
            recs = pick(recs)
        elif mode == "per_gene":
            recs = pick(recs, per_gene=True)
        rows.extend(recs)
    by_term = {}
    for r in rows:
        for t in r["consequences"]:
            by_term[t] = by_term.get(t, 0) + 1
    return RichResult(payload={
        "estimate": rows,
        "annotations": rows,
        "n_variants": len(vs),
        "n_annotations": len(rows),
        "consequence_counts": by_term,
        "mode": mode,
        "method": ("Ensembl Variant Effect Predictor (McLaren et al. "
                   "2016): per-transcript consequence predicates with "
                   "Sequence Ontology terms, severity from Ensembl's "
                   "published consequence table"),
        "note": ("regulatory, TFBS, NMD, miRNA and structural-variant "
                 "terms are in the rank table for severity comparison "
                 "but are not predicted here, since they need a "
                 "regulatory build or transcript flags a gene model does "
                 "not carry"),
    })


vepannotation = vep_annotation


def cheatsheet():
    return ("vepan: Ensembl VEP (McLaren et al. 2016). One record per "
            "(variant, transcript): predicate functions assign Sequence "
            "Ontology consequence terms, ranked 1-41 by Ensembl's "
            "published severity table, so most_severe_consequence and "
            "impact mean something. mode='pick' applies Table 7's order "
            "-- canonical transcript first, then protein coding, then "
            "severity -- and mode='per_gene' does the same per gene. "
            "HGVS c. and p. are emitted, with indels shifted to their "
            "most 3' position.")
