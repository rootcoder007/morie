# morie.fn -- function file (rootcoder007/morie)
r"""Variant calling as image classification, DeepVariant style.

**The pipeline.** Find candidate variants in aligned reads with **high
sensitivity and low specificity**; encode the reference and read data
around each candidate as a pileup image; hand the image to a
classifier that emits probabilities for the three diploid genotypes
(homozygous reference, heterozygous, homozygous alternate).

The asymmetry in step one is deliberate and is the part most often
got wrong. The candidate stage is *supposed* to over-call: for Ion
Torrent data the paper's candidates have a positive predictive value
of **8.1%**, which the classifier then lifts to **99.7%**. Filtering
hard at the candidate stage would throw away true variants that the
classifier could have rescued, and sensitivity lost there is lost for
good -- across datasets the final calls give up a mean of only
**2.3%** of candidate sensitivity.

**What the pileup image is for.** Presenting every read at a locus in
one image lets a convolutional network account for the dependence
*between* reads, rather than treating each as independent evidence the
way a hand-built likelihood does. The paper's argument is that the
network approximates the true but unknown interdependent likelihood
function, and its calibration curve is the evidence.

**What this module implements, and what it does not.** Candidate
generation, the pileup encoding, the genotype posterior from a
supplied scorer, and the evaluation arithmetic are all here and
anchored. The Inception-v2 network itself is **not** reimplemented:
the preprint gives the architecture by reference and does not specify
the per-channel image encoding beyond "reference and read bases,
quality scores, and other read features ... encoded into an RGB pileup
image". ``encode_pileup`` therefore takes the channel set as an
argument and names its default in the result rather than pretending a
specific encoding came from the paper. ``genotype_posterior`` accepts
any scorer, so a trained model can be dropped in.

References
----------
Poplin, R., Newburger, D., Dijamco, J., Nguyen, N., Loy, D., Gross,
S. S., McLean, C. Y. & DePristo, M. A. (2016) "Creating a universal
SNP and small indel variant caller with deep neural networks",
bioRxiv 092890, doi:10.1101/092890 (published as Poplin et al. (2018)
*Nature Biotechnology* 36(10), 983-987,
doi:10.1038/nbt.4235). The high-sensitivity/low-specificity candidate
stage, the pileup image of reference and read data around each
candidate, the Inception-v2 classifier emitting the three diploid
genotype probabilities, the calibration argument for why an image of
all reads captures inter-read dependence, and the printed accounting:
candidate PPV 8.1% raised to 99.7% on Ion Torrent, a mean loss of 2.3%
of candidate sensitivity, and the SOLiD figures of 13.9% PPV at 96.2%
sensitivity.

Li, H. (2011) "A statistical framework for SNP calling, mutation
discovery, association mapping and population genetical parameter
estimation from sequencing data", *Bioinformatics* 27(21), 2987-2993,
doi:10.1093/bioinformatics/btr509, for the conventional genotype
likelihood this replaces.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["GENOTYPES", "CHANNEL_SETS", "pileup_column",
           "find_candidates", "encode_pileup", "genotype_posterior",
           "call_variants", "evaluate"]

GENOTYPES = ("hom_ref", "het", "hom_alt")
CHANNEL_SETS = ("base_quality_strand",)


def pileup_column(reads, position, reference):
    r"""Bases, qualities, strands and mapping qualities at one site."""
    obs = []
    for r in reads:
        start = int(r["pos"])
        seq = r["seq"]
        if start <= position < start + len(seq):
            i = position - start
            obs.append({
                "base": seq[i],
                "bq": (r["bq"][i] if "bq" in r else 30),
                "mq": int(r.get("mq", 60)),
                "reverse": bool(r.get("reverse", False)),
            })
    ref = reference[position] if position < len(reference) else "N"
    return {"observations": obs, "reference": ref, "depth": len(obs)}


def find_candidates(reads, reference, min_alt_count=2,
                    min_alt_fraction=0.05, min_bq=10):
    r"""The deliberately permissive candidate stage.

    Low thresholds are the point: a variant missed here can never be
    recovered downstream, whereas a false candidate is only a
    classification the model has to reject.
    """
    if min_alt_fraction < 0.0 or min_alt_fraction > 1.0:
        raise ValueError("varcal: min_alt_fraction must lie in [0, 1]")
    if min_alt_count < 1:
        raise ValueError("varcal: min_alt_count must be at least 1")
    out = []
    for pos in range(len(reference)):
        col = pileup_column(reads, pos, reference)
        kept = [o for o in col["observations"] if o["bq"] >= min_bq]
        if not kept:
            continue
        counts = {}
        for o in kept:
            counts[o["base"]] = counts.get(o["base"], 0) + 1
        ref = col["reference"]
        for base, n in sorted(counts.items()):
            if base == ref:
                continue
            frac = n / float(len(kept))
            if n >= min_alt_count and frac >= min_alt_fraction:
                out.append({"position": pos, "reference": ref,
                            "alternate": base, "alt_count": n,
                            "depth": len(kept), "alt_fraction": frac})
    return out


def encode_pileup(reads, reference, candidate, width=21, height=100,
                  channels="base_quality_strand"):
    r"""The pileup image around a candidate.

    Rows are reads, columns are reference positions, and each cell
    carries the channel values. The preprint does not specify the
    per-channel mapping, so the set used is named in the result rather
    than presented as the paper's.
    """
    if channels not in CHANNEL_SETS:
        raise ValueError("varcal: channels must be one of %s, got %r"
                         % (", ".join(CHANNEL_SETS), channels))
    if width % 2 == 0:
        raise ValueError("varcal: width must be odd so the candidate "
                         "sits in the middle column")
    half = width // 2
    centre = int(candidate["position"])
    lo, hi = centre - half, centre + half
    rows = []
    for r in reads:
        start = int(r["pos"])
        stop = start + len(r["seq"])
        if stop <= lo or start > hi:
            continue
        row = []
        for p in range(lo, hi + 1):
            if start <= p < stop and 0 <= p < len(reference):
                i = p - start
                b = r["seq"][i]
                row.append((_base_code(b),
                            min(float(r["bq"][i] if "bq" in r else 30)
                                / 60.0, 1.0),
                            0.0 if r.get("reverse") else 1.0,
                            1.0 if b == reference[p] else 0.0))
            else:
                row.append((0.0, 0.0, 0.0, 0.0))
        rows.append(row)
        if len(rows) >= height:
            break
    ref_row = [(_base_code(reference[p]) if 0 <= p < len(reference)
                else 0.0, 1.0, 1.0, 1.0) for p in range(lo, hi + 1)]
    return {"reference_row": ref_row, "read_rows": rows,
            "n_reads": len(rows), "width": width,
            "centre": centre,
            "channels": ("base", "base_quality", "strand",
                         "matches_reference"),
            "channel_set": channels,
            "note": "the preprint specifies an RGB pileup image of "
                    "bases, qualities and read features but not the "
                    "per-channel mapping; this set is named here "
                    "rather than attributed to the paper"}


def _base_code(b):
    return {"A": 0.25, "C": 0.5, "G": 0.75, "T": 1.0}.get(
        str(b).upper(), 0.0)


def genotype_posterior(image, scorer=None, prior=None):
    r"""Probabilities over the three diploid genotypes.

    ``scorer`` maps an image to three non-negative scores; the default
    is a transparent pileup-fraction rule so the pipeline runs without
    a trained network, and it is labelled as such in the result.
    """
    if prior is None:
        prior = (0.9985, 0.001, 0.0005)
    if len(prior) != 3 or abs(sum(prior) - 1.0) > 1e-9:
        raise ValueError("varcal: the prior must be three "
                         "probabilities summing to 1")
    if scorer is None:
        alt = 0.0
        tot = 0.0
        mid = image["width"] // 2
        for row in image["read_rows"]:
            cell = row[mid]
            if cell[0] > 0.0:
                tot += 1.0
                alt += 0.0 if cell[3] > 0.5 else 1.0
        f = alt / tot if tot > 0.0 else 0.0
        scores = (max(1.0 - 2.0 * f, 0.0),
                  1.0 - abs(2.0 * f - 1.0),
                  max(2.0 * f - 1.0, 0.0))
        source = "pileup-fraction fallback, NOT a trained network"
    else:
        scores = tuple(float(v) for v in scorer(image))
        if len(scores) != 3 or any(v < 0.0 for v in scores):
            raise ValueError("varcal: the scorer must return three "
                             "non-negative scores")
        source = "supplied scorer"
    post = [scores[i] * prior[i] for i in range(3)]
    tot = sum(post)
    if tot <= 0.0:
        post = list(prior)
        tot = 1.0
    post = [v / tot for v in post]
    k = max(range(3), key=lambda i: post[i])
    return {"posterior": dict(zip(GENOTYPES, post)),
            "call": GENOTYPES[k], "quality": _phred(1.0 - post[k]),
            "scores": scores, "source": source}


def _phred(p):
    p = max(min(float(p), 1.0), 1e-12)
    return -10.0 * math.log10(p)


def call_variants(reads, reference, scorer=None, min_quality=10.0,
                  **kw):
    r"""Candidates, images, posteriors, calls."""
    cands = find_candidates(reads, reference, **kw)
    calls = []
    for c in cands:
        img = encode_pileup(reads, reference, c)
        g = genotype_posterior(img, scorer)
        calls.append({**c, **g,
                      "passes": g["call"] != "hom_ref"
                      and g["quality"] >= min_quality})
    return RichResult(payload={
        "estimate": sum(1 for c in calls if c["passes"]),
        "candidates": cands, "n_candidates": len(cands),
        "calls": calls,
        "n_called": sum(1 for c in calls if c["passes"]),
        "method": "candidate generation, pileup encoding and "
                  "genotype classification; Poplin et al. (2016)",
    })


def evaluate(called, truth, candidates=None):
    r"""PPV and sensitivity, and what the classifier stage bought.

    With the candidate set supplied, the result also reports the
    candidate-stage figures, which is where the paper's 8.1% to 99.7%
    comparison lives.
    """
    tset = {(t["position"], t["alternate"]) for t in truth}
    cset = {(c["position"], c["alternate"]) for c in called}
    tp = len(cset & tset)
    ppv = tp / float(len(cset)) if cset else 0.0
    sens = tp / float(len(tset)) if tset else 0.0
    out = {"true_positives": tp, "called": len(cset),
           "truth": len(tset), "ppv": ppv, "sensitivity": sens}
    if candidates is not None:
        aset = {(c["position"], c["alternate"]) for c in candidates}
        atp = len(aset & tset)
        out["candidate_ppv"] = (atp / float(len(aset)) if aset
                                else 0.0)
        out["candidate_sensitivity"] = (atp / float(len(tset))
                                        if tset else 0.0)
        out["ppv_gain"] = ppv - out["candidate_ppv"]
        out["sensitivity_loss"] = out["candidate_sensitivity"] - sens
    return out


def cheatsheet():
    return ("varcal: candidates are generated with HIGH sensitivity "
            "and low specificity on purpose -- 8.1% PPV on Ion "
            "Torrent, which the classifier lifts to 99.7% while "
            "giving up a mean 2.3% of candidate sensitivity. The "
            "pileup image puts every read at the locus in one picture "
            "so the network can use the dependence between reads. The "
            "Inception-v2 network itself is not reimplemented here; "
            "genotype_posterior takes any scorer, and the default is "
            "a labelled fallback, not a trained model.")


# compact alias per ledger/NAMING.md
deep_variant_call = call_variants
