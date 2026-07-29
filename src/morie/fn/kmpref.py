# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefix tuning: learned key/value vectors prepended at every
layer."""

import numpy as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["kamath_prefix_tuning"]


def kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input, Q=None):
    """K = [K_prefix; K_input], V = [V_prefix; V_input]; only the
    prefixes are trained.

    Pass ``Q`` to see what the extended attention actually computes --
    that step is DELEGATED to ``morie.fn.attsdp``, the package's one
    scaled-dot-product implementation, instead of a second copy here.
    The prefix rows come FIRST, which matters for any positional or
    causal masking the caller applies downstream.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, prefix tuning
    (Li and Liang 2021).

    Examples
    --------
    >>> out = kamath_prefix_tuning([[1.0, 0.0]], [[9.0]],
    ...                            [[0.0, 1.0]], [[1.0]])
    >>> out["K"]
    [[1.0, 0.0], [0.0, 1.0]]
    >>> out["prefix_len"], out["n_trainable"]
    (1, 3)
    >>> att = kamath_prefix_tuning([[1.0, 0.0]], [[9.0]],
    ...     [[0.0, 1.0]], [[1.0]], Q=[[1.0, 0.0]])
    >>> import math
    >>> w = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    >>> abs(att["attention_output"][0][0] - (9 * w + 1 * (1 - w))) < 1e-12
    True
    """
    PK = np.atleast_2d(np.asarray(prefix_K, dtype=float))
    PV = np.atleast_2d(np.asarray(prefix_V, dtype=float))
    K = np.atleast_2d(np.asarray(K_input, dtype=float))
    V = np.atleast_2d(np.asarray(V_input, dtype=float))
    if PK.shape[0] != PV.shape[0]:
        raise ValueError(
            f"the key prefix has {PK.shape[0]} rows and the value "
            f"prefix {PV.shape[0]}; they are the same virtual tokens.")
    if PK.shape[0] == 0:
        raise ValueError("an empty prefix tunes nothing.")
    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"{K.shape[0]} input keys but {V.shape[0]} input values.")
    if PK.shape[1] != K.shape[1]:
        raise ValueError(
            f"key prefix width {PK.shape[1]} != input key width "
            f"{K.shape[1]}.")
    if PV.shape[1] != V.shape[1]:
        raise ValueError(
            f"value prefix width {PV.shape[1]} != input value width "
            f"{V.shape[1]}.")
    Kf = np.vstack([PK, K])
    Vf = np.vstack([PV, V])
    payload = {
        "K": [[float(v) for v in row] for row in Kf],
        "V": [[float(v) for v in row] for row in Vf],
        "prefix_len": int(PK.shape[0]),
        "seq_len": int(Kf.shape[0]),
        "n_trainable": int(PK.size + PV.size),
        "estimate": int(PK.shape[0]),
        "n": int(Kf.shape[0]),
        "method": "Prefix tuning key/value concatenation"}
    if Q is not None:
        att = scaled_dot_product_attention(Q, Kf, Vf)
        payload["attention_output"] = att["output"]
        payload["attention_weights"] = att["attention"]
        payload["prefix_attention_mass"] = [
            float(sum(row[:PK.shape[0]])) for row in att["attention"]]
        payload["estimate"] = float(att["estimate"])
        payload["method"] += " + attsdp attention"
    return RichResult(payload=payload)


def cheatsheet():
    return "kmpref: [P_K; K], [P_V; V]; optional Q runs attsdp attention"
