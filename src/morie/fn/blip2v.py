# morie.fn -- function file (rootcoder007/morie)
r"""BLIP-2: bridging modalities with everything else frozen.

Vision-language pre-training gets expensive because it trains
everything end to end. BLIP-2's premise is that the expensive parts
already exist: pre-trained vision models give high-quality visual
representations and large language models give strong generation. What
is missing is the bridge, so **freeze both** and train only a
lightweight **Querying Transformer** between them.

**The Q-Former is a fixed set of learnable queries, and the size of
that set is the bottleneck.** A frozen image encoder emits hundreds of
patch features; the Q-Former's :math:`N` queries (32 in the paper)
attend to them and emit :math:`N` vectors. So the visual input handed
to the language model has a *fixed, small* width regardless of image
resolution -- which is what makes it affordable to feed a frozen LLM
and forces the queries to extract the language-relevant content rather
than pass everything through.

**Two stages, because the two gaps are different.** The first stage
bootstraps vision-language *representation* learning from the frozen
image encoder, training the queries to extract text-relevant visual
features. The second bootstraps vision-to-language *generative*
learning from the frozen LLM, projecting the query outputs into its
input space. Doing the second without the first leaves the queries
extracting features the language model cannot use.

**The economics are the claim.** The paper reports outperforming
Flamingo80B by 8.7% on zero-shot VQAv2 with **54x fewer trainable
parameters** -- and ``trainable_fraction`` computes that ratio, since
the whole argument is about what is *not* trained.

References
----------
Li, J., Li, D., Savarese, S. & Hoi, S. (2023) "BLIP-2: Bootstrapping
Language-Image Pre-training with Frozen Image Encoders and Large
Language Models", *Proceedings of the 40th International Conference on
Machine Learning (ICML 2023)*, PMLR 202, 19730-19742,
arXiv:2301.12597. The abstract and Sec. 3: a lightweight Querying
Transformer following a TWO-STAGE strategy to bridge the modality gap;
the first stage bootstrapping vision-language representation learning
from a frozen image encoder and the second bootstrapping
vision-to-language generative learning from a frozen LLM, enabling
zero-shot instructed image-to-text generation; a compute-efficient
method bootstrapping from off-the-shelf pre-trained vision and
language models; and state-of-the-art results with significantly fewer
trainable parameters, outperforming Flamingo80B by 8.7% on zero-shot
VQAv2 with 54x fewer trainable parameters.

Alayrac, J.-B. et al. (2022) "Flamingo: a Visual Language Model for
Few-Shot Learning", *NeurIPS 2022*, arXiv:2204.14198. The baseline
the parameter comparison is against.

Radford, A. et al. (2021) "Learning Transferable Visual Models From
Natural Language Supervision", *ICML 2021*, PMLR 139, 8748-8763,
arXiv:2103.00020. The frozen image encoders this builds on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["query_tokens", "qformer_attend", "trainable_fraction",
           "stage_one_objectives", "project_to_llm"]

_EPS = 1e-12
_STAGES = (1, 2)


def query_tokens(n_queries, dim, seed=0, scale=0.02):
    r"""The learnable queries -- a FIXED, small number of them.

    Their count, not the image resolution, sets the width of the
    visual input the language model receives.
    """
    n, d = int(n_queries), int(dim)
    if n < 1 or d < 1:
        raise ValueError("blip2v: the query count and dimension must "
                         "be positive")
    rng = np.random.default_rng(seed)
    return [[(float(rng.uniform()) - 0.5) * 2.0 * scale
             for _ in range(d)] for _ in range(n)]


def qformer_attend(queries, image_features, WQ, WK, WV):
    r"""Cross-attention from the queries to the frozen patch features.

    Output width equals the QUERY count, so a 224-pixel and a
    1024-pixel image hand the language model the same number of
    vectors.
    """
    Q = [[float(v) for v in r] for r in k.mat(queries)]
    F = [[float(v) for v in r] for r in k.mat(image_features)]
    dk = len(WQ)

    def proj(W, x):
        return [sum(W[o][j] * x[j] for j in range(len(x)))
                for o in range(len(W))]

    out, weights = [], []
    for q in Q:
        qq = proj(WQ, q)
        sc = [sum(qq[a] * proj(WK, f)[a] for a in range(dk))
              / math.sqrt(dk) for f in F]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        weights.append(w)
        vs = [proj(WV, f) for f in F]
        out.append([sum(w[j] * vs[j][a] for j in range(len(F)))
                    for a in range(len(vs[0]))])
    return {"output": out, "weights": weights,
            "n_queries": len(Q), "n_patches": len(F),
            "compression": len(F) / float(len(Q)),
            "note": "the output width is the QUERY count, whatever "
                    "the image resolution"}


def trainable_fraction(qformer_params, frozen_vision_params,
                       frozen_llm_params):
    r"""What fraction is actually trained.

    The whole argument is about what is NOT trained, so the ratio is
    the headline number rather than a footnote.
    """
    q = float(qformer_params)
    tot = q + float(frozen_vision_params) + float(frozen_llm_params)
    if tot <= 0.0:
        raise ValueError("blip2v: the parameter counts must be "
                         "positive")
    return {"trainable": q, "total": tot, "fraction": q / tot,
            "frozen_fraction": 1.0 - q / tot,
            "note": "vision encoder and language model both frozen"}


def stage_one_objectives(query_out, text_out, temperature=0.07):
    r"""Stage 1: align the queries with text before any generation.

    Contrastive image-text alignment on the query outputs. Skipping
    this and going straight to stage 2 leaves the queries extracting
    features the language model cannot use.
    """
    Q = [[float(v) for v in r] for r in k.mat(query_out)]
    T = [float(v) for v in k.vec(text_out)]
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("blip2v: the temperature must be positive")

    def cos(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na <= _EPS or nb <= _EPS:
            raise ValueError("blip2v: a zero embedding has no "
                             "direction")
        return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)

    sims = [cos(q, T) for q in Q]
    return {"per_query_similarity": sims,
            "image_text_similarity": max(sims),
            "best_query": max(range(len(sims)),
                              key=lambda i: sims[i]),
            "logit": max(sims) / t,
            "note": "the image-text score is the MAXIMUM over "
                    "queries, not the mean -- one query may carry the "
                    "relevant content"}


def project_to_llm(query_out, W, b=None):
    r"""Stage 2: project the query outputs into the frozen LLM's input
    space."""
    Q = [[float(v) for v in r] for r in k.mat(query_out)]
    d_out = len(W)
    bb = [0.0] * d_out if b is None else [float(v) for v in k.vec(b)]
    out = []
    for q in Q:
        if len(W[0]) != len(q):
            raise ValueError("blip2v: the projection expects %d "
                             "inputs but the query output is %d"
                             % (len(W[0]), len(q)))
        out.append([bb[o] + sum(W[o][j] * q[j]
                                for j in range(len(q)))
                    for o in range(d_out)])
    return RichResult(payload={
        "estimate": out, "soft_prompt": out,
        "n_tokens": len(out), "dim": d_out,
        "method": "BLIP-2 two-stage bridging; Li, Li, Savarese & Hoi "
                  "(2023)",
        "note": "the projected queries act as a soft prompt prefixed "
                "to the frozen LLM's input",
    })


def cheatsheet():
    return ("blip2v: the expensive parts already exist -- FREEZE the "
            "image encoder and the LLM and train only a lightweight "
            "Q-Former between them. A FIXED small set of learnable "
            "queries (32) cross-attends to hundreds of patch features, "
            "so the visual input handed to the language model has "
            "constant width whatever the resolution, and the queries "
            "must EXTRACT rather than pass through. TWO stages, "
            "because the gaps differ: representation alignment from "
            "the frozen encoder, then generative learning into the "
            "frozen LLM. The claim is economic: beating Flamingo80B "
            "with 54x fewer trainable parameters.")


# compact alias per ledger/NAMING.md
blip2 = qformer_attend

# public names resolved by fn/_lazy_map.json
blip2_qformer = qformer_attend
blip2qformer = qformer_attend
