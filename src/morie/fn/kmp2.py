# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P-tuning v2: a learnable prefix at EVERY layer, not just the
input."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_p_tuning_v2"]


def kamath_p_tuning_v2(prefixes_by_layer, inputs_by_layer):
    """At each layer l: K_l = [P_K_l; K_l^in], V_l = [P_V_l; V_l^in];
    only P_K_l and P_V_l are trained.

    ``prefixes_by_layer`` and ``inputs_by_layer`` are per-layer
    ``(K, V)`` pairs. The point of v2 over prompt tuning is depth, so
    a prefix count that does not match the layer count is an error,
    not something to broadcast: a prefix quietly reused at every layer
    is p-tuning v1 wearing v2's name.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, P-tuning v2.

    Examples
    --------
    >>> pre = [([[1.0, 1.0]], [[2.0, 2.0]])]
    >>> inp = [([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [4.0, 4.0]])]
    >>> out = kamath_p_tuning_v2(pre, inp)
    >>> out["K"][0]
    [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0]]
    >>> out["prefix_len"], out["n_layers"]
    ([1], 1)
    >>> out["n_trainable"]
    4
    """
    pre = list(prefixes_by_layer)
    inp = list(inputs_by_layer)
    if not pre:
        raise ValueError("no prefixes supplied.")
    if len(pre) != len(inp):
        raise ValueError(
            f"{len(pre)} prefix layers for {len(inp)} input layers; "
            "p-tuning v2 needs one prefix per layer.")
    Ks, Vs, plens = [], [], []
    trainable = 0
    for l, (p_pair, i_pair) in enumerate(zip(pre, inp)):
        if len(p_pair) != 2 or len(i_pair) != 2:
            raise ValueError(
                f"layer {l}: expected (K, V) pairs on both sides.")
        PK = np.atleast_2d(np.asarray(p_pair[0], dtype=float))
        PV = np.atleast_2d(np.asarray(p_pair[1], dtype=float))
        K = np.atleast_2d(np.asarray(i_pair[0], dtype=float))
        V = np.atleast_2d(np.asarray(i_pair[1], dtype=float))
        if PK.shape[0] != PV.shape[0]:
            raise ValueError(
                f"layer {l}: the key prefix has {PK.shape[0]} rows and "
                f"the value prefix {PV.shape[0]}; they are the same "
                "virtual tokens.")
        if K.shape[0] != V.shape[0]:
            raise ValueError(
                f"layer {l}: {K.shape[0]} input keys but {V.shape[0]} "
                "input values.")
        if PK.shape[1] != K.shape[1]:
            raise ValueError(
                f"layer {l}: key prefix width {PK.shape[1]} != input key "
                f"width {K.shape[1]}.")
        if PV.shape[1] != V.shape[1]:
            raise ValueError(
                f"layer {l}: value prefix width {PV.shape[1]} != input "
                f"value width {V.shape[1]}.")
        Ks.append([[float(v) for v in row] for row in np.vstack([PK, K])])
        Vs.append([[float(v) for v in row] for row in np.vstack([PV, V])])
        plens.append(int(PK.shape[0]))
        trainable += int(PK.size + PV.size)
    return RichResult(payload={
        "K": Ks, "V": Vs, "prefix_len": plens,
        "n_layers": len(pre), "n_trainable": trainable,
        "estimate": trainable, "n": len(pre),
        "method": "P-tuning v2 deep prefix concatenation"})


def cheatsheet():
    return "kmp2: [P_K_l; K_l] and [P_V_l; V_l] at every layer"
