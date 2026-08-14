# morie.fn -- function file (rootcoder007/morie)
r"""LLaVA: visual instruction tuning.

Instruction-following data improves zero-shot behaviour on new tasks
in language, and the idea had been little explored for multimodal
models. The obstacle is data: nobody had image-grounded instruction
data at scale.

**The move is to generate it with a language-only model.** GPT-4,
which cannot see images, is given *symbolic* representations of an
image -- captions and object bounding boxes -- and asked to produce
conversations, detailed descriptions and complex reasoning about it.
The image never enters the generator; the text describing it does.
That is what makes the pipeline possible at all, and it is also its
main limitation, since anything absent from the captions and boxes
cannot appear in the generated instruction.

**The architecture is deliberately thin.** A vision encoder, a
**single projection matrix** mapping its features into the language
model's word-embedding space, and the language model. The projected
patches are treated as tokens, so no cross-attention layers are added
anywhere. That thinness is a claim: most of the capability is already
present in the two pre-trained parts.

**Two stages with different parts frozen.** First the projection alone
is trained on image-caption pairs, aligning the two spaces while
everything else is frozen. Then the projection *and* the language
model are tuned on the generated instruction data. Reversing the order
tunes the language model against features that do not yet mean
anything.

The reported results: 85.1% relative to GPT-4 on a synthetic
multimodal instruction-following dataset, and 92.53% on Science QA
when combined with GPT-4.

References
----------
Liu, H., Li, C., Wu, Q. & Lee, Y. J. (2023) "Visual Instruction
Tuning", *Advances in Neural Information Processing Systems 36
(NeurIPS 2023)*, arXiv:2304.08485. The abstract: instruction tuning
improves zero-shot capabilities on new tasks but is less explored in
the multimodal field; the first attempt to use LANGUAGE-ONLY GPT-4 to
generate multimodal language-image instruction-following data; LLaVA
as an end-to-end trained large multimodal model connecting a vision
encoder and an LLM for general-purpose visual and language
understanding; two evaluation benchmarks constructed for visual
instruction following; a 85.1% relative score compared with GPT-4 on a
synthetic multimodal instruction-following dataset; and 92.53%
accuracy on Science QA from the synergy of LLaVA and GPT-4.

Radford, A. et al. (2021) "Learning Transferable Visual Models From
Natural Language Supervision", *ICML 2021*, PMLR 139, 8748-8763,
arXiv:2103.00020. The vision encoder.

Li, J., Li, D., Savarese, S. & Hoi, S. (2023) "BLIP-2", *ICML 2023*,
PMLR 202, 19730-19742, arXiv:2301.12597. The alternative bridge, a
Q-Former rather than a projection; implemented in :mod:`blip2v`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["symbolic_representation", "instruction_prompt",
           "project_patches", "build_sequence", "training_stage"]

_EPS = 1e-12
_KINDS = ("conversation", "detailed_description", "complex_reasoning")


def symbolic_representation(captions, boxes):
    r"""The image as TEXT, which is all the generator ever sees.

    Captions and bounding boxes. Anything they omit cannot appear in
    the generated instruction -- the pipeline's ceiling, stated.
    """
    caps = [str(v) for v in captions]
    bx = list(boxes)
    if not caps and not bx:
        raise ValueError("llavx: an image with no captions and no "
                         "boxes has no symbolic representation")
    lines = list(caps)
    for (name, x, y, w, h) in bx:
        lines.append("%s: [%.3f, %.3f, %.3f, %.3f]"
                     % (str(name), float(x), float(y), float(w),
                        float(h)))
    return {"text": "\n".join(lines), "n_captions": len(caps),
            "n_boxes": len(bx),
            "note": "the generator is LANGUAGE-ONLY; the image itself "
                    "never reaches it"}


def instruction_prompt(symbolic, kind="conversation"):
    r"""Ask the language-only model for one of the three data types."""
    if kind not in _KINDS:
        raise ValueError("llavx: kind must be one of %s, got %r"
                         % (", ".join(_KINDS), kind))
    ask = {"conversation": "Ask and answer questions about this "
                           "image as if you can see it.",
           "detailed_description": "Describe this image in detail.",
           "complex_reasoning": "Give a question requiring "
                                "step-by-step reasoning about this "
                                "image, and answer it."}[kind]
    return {"prompt": "%s\n\n%s" % (symbolic["text"], ask),
            "kind": kind}


def project_patches(patch_features, W, b=None):
    r"""ONE matrix into the word-embedding space.

    No cross-attention is added anywhere: the projected patches are
    simply tokens.
    """
    F = [[float(v) for v in r] for r in k.mat(patch_features)]
    d_out = len(W)
    if len(W[0]) != len(F[0]):
        raise ValueError("llavx: the projection expects %d features "
                         "but got %d" % (len(W[0]), len(F[0])))
    bb = [0.0] * d_out if b is None else [float(v) for v in k.vec(b)]
    return [[bb[o] + sum(W[o][j] * f[j] for j in range(len(f)))
             for o in range(d_out)] for f in F]


def build_sequence(visual_tokens, text_embeddings):
    r"""Visual tokens then text, in one sequence.

    The language model sees no distinction, which is the point of
    projecting into its own embedding space.
    """
    V = [[float(v) for v in r] for r in k.mat(visual_tokens)]
    T = [[float(v) for v in r] for r in k.mat(text_embeddings)]
    if V and T and len(V[0]) != len(T[0]):
        raise ValueError("llavx: visual tokens are %d-dimensional but "
                         "text embeddings are %d -- the projection "
                         "target is wrong" % (len(V[0]), len(T[0])))
    return RichResult(payload={
        "estimate": V + T, "sequence": V + T,
        "n_visual": len(V), "n_text": len(T),
        "method": "visual instruction tuning; Liu, Li, Wu & Lee "
                  "(2023)",
        "note": "projected patches ARE tokens -- no cross-attention "
                "layers are introduced",
    })


def training_stage(stage):
    r"""Which parameters move at each stage.

    Stage 1 aligns the projection while everything else is frozen;
    stage 2 tunes projection and language model on the generated
    instructions. The order matters.
    """
    s = int(stage)
    if s not in (1, 2):
        raise ValueError("llavx: the stage must be 1 or 2, got %r"
                         % (stage,))
    if s == 1:
        return {"stage": 1, "trainable": ["projection"],
                "frozen": ["vision_encoder", "language_model"],
                "data": "image-caption pairs",
                "note": "align the spaces before tuning anything on "
                        "them"}
    return {"stage": 2, "trainable": ["projection", "language_model"],
            "frozen": ["vision_encoder"],
            "data": "GPT-4 generated instruction-following data",
            "note": "tuning the language model first would tune it "
                    "against features that do not yet mean anything"}


def cheatsheet():
    return ("llavx: instruction tuning works in language and lacked "
            "MULTIMODAL data, so generate it with a LANGUAGE-ONLY "
            "GPT-4 fed a SYMBOLIC image -- captions and boxes. The "
            "image never reaches the generator, which is what makes "
            "the pipeline possible and also caps it: what the captions "
            "omit cannot be asked about. Architecture is deliberately "
            "thin: ONE projection matrix into the word-embedding "
            "space, projected patches used as tokens, no "
            "cross-attention. Stage 1 trains only the projection; "
            "stage 2 adds the language model.")


# compact alias per ledger/NAMING.md
visualinstruction = build_sequence

# public names resolved by fn/_lazy_map.json
llava_visual_chat = build_sequence
