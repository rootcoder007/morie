# morie.fn -- function file (rootcoder007/morie)
r"""TimesFM: a decoder-only foundation model with input patching.

One model, pretrained once, applied zero-shot to datasets it has never
seen -- across domains, history lengths, prediction lengths and time
granularities. Two things make that possible, and the second is the
architectural one.

**A decoder-only stack with input patching.** The history is cut into
patches, each patch becomes a token, and a causal decoder attends over
them. Decoder-only rather than encoder-decoder because forecasting is
naturally autoregressive over patches: every patch predicts what comes
next, so a single training sequence of :math:`N` patches supplies
:math:`N` training signals rather than one.

**The asymmetry that matters: the output patch may be longer than the
input patch.** This is the design decision with real consequences. If
the input patch is :math:`p` and the output patch is :math:`q`, a
horizon :math:`H` needs :math:`\lceil H/q \rceil` autoregressive steps,
not :math:`\lceil H/p \rceil`. Making :math:`q > p` cuts the number of
generation steps, and with it the error accumulation that makes long
autoregressive rollouts drift. Setting :math:`q = p` recovers the
symmetric case, and setting :math:`q \ge H` makes the forecast a
**single** step -- direct multi-horizon prediction with no rollout at
all. All three are reachable from the same model, which is what lets
one checkpoint serve many horizons.

**Scale, in context.** 200M parameters and O(100B) timepoints -- tiny
next to a large language model, and the paper's point is that this is
enough. A model trained from scratch on time series beats prompting
GPT-3 or Llama-2 as a zero-shot forecaster, at a small fraction of the
cost. The corpus is real data (web search interest, Wikipedia page
views) plus synthetic series generated to cover shapes the real data
misses.

**What this module implements.** The patching contract and the rollout
arithmetic -- the parts that are architecture rather than weights.
There are no pretrained parameters here: ``rollout`` takes a predictor
callable, so the accounting can be checked exactly without a 200M-
parameter checkpoint, and the anchor uses that to verify the step
counts and the error-accumulation behaviour.

References
----------
Das, A., Kong, W., Sen, R. & Zhou, Y. (2024) "A decoder-only
foundation model for time-series forecasting", *Proceedings of the
41st International Conference on Machine Learning*, PMLR 235,
arXiv:2310.10688. The two key elements (a large-scale corpus of real
and synthetic series, and a decoder-style attention architecture with
input patching), the 200M parameter and O(100B) timepoint scale, the
claim that a from-scratch time-series model beats LLM prompting at a
fraction of the cost, and operation across differing history lengths,
prediction lengths and granularities at inference time.

Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. (2023) "A Time
Series is Worth 64 Words: Long-term Forecasting with Transformers",
*ICLR 2023*, arXiv:2211.14730. The patching idea, used here with the
input/output asymmetry added.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["input_patches", "causal_mask", "rollout_steps", "rollout",
           "horizon_plan"]

_EPS = 1e-12


def input_patches(x, patch_len, pad_value=0.0):
    r"""Cut the history into input patches, left-padding if needed.

    Padding on the **left** keeps the most recent observation at the
    end of the last patch, which is what a causal decoder needs.
    """
    v = [float(q) for q in k.vec(x)]
    p = int(patch_len)
    if p < 1:
        raise ValueError("timesfm: patch_len must be at least 1")
    if not v:
        raise ValueError("timesfm: the history is empty")
    rem = len(v) % p
    pad = (p - rem) % p
    padded = [float(pad_value)] * pad + v
    n = len(padded) // p
    return {"patches": [padded[i * p:(i + 1) * p] for i in range(n)],
            "n_patches": n, "patch_len": p, "n_padded": pad,
            "L": len(v),
            "note": "padded on the LEFT so the newest point ends the "
                    "final patch"}


def causal_mask(n_patches):
    r"""Lower-triangular mask: patch :math:`i` sees patches
    :math:`\le i`.

    Every patch position is a training signal, which is why a
    decoder-only stack extracts :math:`N` supervised examples from one
    sequence of :math:`N` patches.
    """
    n = int(n_patches)
    if n < 1:
        raise ValueError("timesfm: need at least one patch")
    return {"mask": [[1.0 if j <= i else 0.0 for j in range(n)]
                     for i in range(n)],
            "n_patches": n,
            "training_signals": n}


def rollout_steps(horizon, output_patch_len):
    r"""Autoregressive steps needed: :math:`\lceil H/q \rceil`."""
    H, q = int(horizon), int(output_patch_len)
    if H < 1:
        raise ValueError("timesfm: the horizon must be at least 1")
    if q < 1:
        raise ValueError("timesfm: output_patch_len must be at least "
                         "1")
    return {"steps": int(math.ceil(H / float(q))), "horizon": H,
            "output_patch_len": q,
            "single_step": q >= H}


def horizon_plan(horizon, input_patch_len, output_patch_len):
    r"""Compare rollout cost when :math:`q > p`, :math:`q = p`, and
    :math:`q \ge H`.

    The whole point of the asymmetry: a longer output patch means
    fewer generation steps and so less accumulated error.
    """
    H = int(horizon)
    p, q = int(input_patch_len), int(output_patch_len)
    return {"steps_asymmetric": rollout_steps(H, q)["steps"],
            "steps_symmetric": rollout_steps(H, p)["steps"],
            "steps_direct": rollout_steps(H, H)["steps"],
            "input_patch_len": p, "output_patch_len": q,
            "horizon": H,
            "speedup_vs_symmetric":
                rollout_steps(H, p)["steps"]
                / float(rollout_steps(H, q)["steps"]),
            "note": "q > p cuts generation steps; q >= H makes the "
                    "forecast a single direct prediction"}


def rollout(history, predictor, horizon, input_patch_len,
            output_patch_len):
    r"""Autoregressive forecast, feeding predictions back as context.

    ``predictor`` maps a list of input patches to the next
    ``output_patch_len`` values. Supplied by the caller so the rollout
    arithmetic is checkable without pretrained weights.
    """
    v = [float(q) for q in k.vec(history)]
    p, q = int(input_patch_len), int(output_patch_len)
    plan = rollout_steps(horizon, q)
    out, ctx = [], list(v)
    for _ in range(plan["steps"]):
        pat = input_patches(ctx, p)["patches"]
        nxt = [float(z) for z in predictor(pat)]
        if len(nxt) != q:
            raise ValueError("timesfm: the predictor returned %d "
                             "values but output_patch_len is %d"
                             % (len(nxt), q))
        out.extend(nxt)
        ctx.extend(nxt)
    return RichResult(payload={
        "estimate": out[:int(horizon)],
        "forecast": out[:int(horizon)],
        "steps": plan["steps"], "horizon": int(horizon),
        "input_patch_len": p, "output_patch_len": q,
        "context_grew_to": len(ctx),
        "method": "decoder-only patched rollout; Das, Kong, Sen & "
                  "Zhou (2024)",
    })


def cheatsheet():
    return ("timesfm: decoder-only + input patching. Causal attention "
            "over patches means N patches give N training signals, "
            "not one. The design choice that matters: the OUTPUT "
            "patch may be LONGER than the input patch, so a horizon H "
            "needs ceil(H/q) generation steps rather than ceil(H/p) "
            "-- fewer rollouts, less accumulated drift. q >= H is a "
            "single direct prediction. 200M parameters and O(100B) "
            "timepoints beats prompting a large language model, at a "
            "fraction of the cost.")


# compact alias per ledger/NAMING.md -- timesf and timesfm are the
# same ledger entry duplicated
timesfmforecast = rollout

# public names resolved by fn/_lazy_map.json
timesfm = rollout
