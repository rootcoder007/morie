# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GoogLeNet/Inception with parallel filter modules."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_googlenet", "inception_module"]

# (name, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj) -- the 2014 paper.
_INCEPTION = [
    ("3a", 64, 96, 128, 16, 32, 32),
    ("3b", 128, 128, 192, 32, 96, 64),
    ("4a", 192, 96, 208, 16, 48, 64),
    ("4b", 160, 112, 224, 24, 64, 64),
    ("4c", 128, 128, 256, 24, 64, 64),
    ("4d", 112, 144, 288, 32, 64, 64),
    ("4e", 256, 160, 320, 32, 128, 128),
    ("5a", 256, 160, 320, 32, 128, 128),
    ("5b", 384, 192, 384, 48, 128, 128),
]
_POOL_AFTER = {"3b", "4e"}


def inception_module(in_ch, o1, r3, o3, r5, o5, pp):
    """Parameter count of one Inception module, branch by branch.

    The 1x1 reductions are the whole trick: a 5x5 convolution straight
    from ``in_ch`` channels would cost ``25 * in_ch * o5``, but routing it
    through ``r5`` channels first costs ``in_ch*r5 + 25*r5*o5``, which for
    the paper's widths is several times cheaper.
    """
    b1 = in_ch * o1 + o1
    b3 = (in_ch * r3 + r3) + (9 * r3 * o3 + o3)
    b5 = (in_ch * r5 + r5) + (25 * r5 * o5 + o5)
    bp = in_ch * pp + pp
    naive5 = 25 * in_ch * o5 + o5
    return {
        "branch_1x1": int(b1),
        "branch_3x3": int(b3),
        "branch_5x5": int(b5),
        "branch_pool": int(bp),
        "out_channels": int(o1 + o3 + o5 + pp),
        "params": int(b1 + b3 + b5 + bp),
        "naive_5x5_params": int(naive5),
        "reduction_saving": int(naive5 - b5),
    }


def geron_googlenet(n_classes=1000, input_size=224, in_channels=3, dropout=0.4):
    """
    GoogLeNet/Inception with parallel filter modules.

    Formula: Inception: concat(1x1, 3x3, 5x5, pool)

    Architecture resolved against a concrete input, in the ``hmalex``
    manner: every stem layer and all nine Inception modules are laid out
    with real spatial sizes and exact parameter counts.

    The two structural ideas are both quantified rather than described.
    Inception concatenates four parallel branches, so a module's output
    width is the *sum* of its branch widths -- and that number then
    becomes the next module's input width, which is why the counts grow
    the way they do. And each expensive branch is preceded by a 1x1
    reduction: ``reduction_saving`` reports, per module, how many
    parameters that bottleneck saves against the naive 5x5.

    The classifier is a global average pool followed by a single linear
    layer, not the two 4096-wide layers of AlexNet, which is how
    GoogLeNet reaches ~6.8M parameters -- roughly a tenth of AlexNet --
    while being deeper.

    Parameters
    ----------
    n_classes : int, default 1000
    input_size : int, default 224
    in_channels : int, default 3
    dropout : float, default 0.4

    Returns
    -------
    result : RichResult
        Keys: layers, modules, total_params, conv_params, fc_params,
        output_shape, final_feature_map, total_reduction_saving,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_googlenet(1000)
    >>> r["total_params"]
    6998552
    >>> r["final_feature_map"]
    (1024, 7, 7)
    >>> r["modules"][0]["name"], r["modules"][0]["out_channels"]
    ('3a', 256)
    >>> r["modules"][0]["params"]
    163696

    The 1x1 reductions save millions of parameters overall:

    >>> r["total_reduction_saving"] > 3000000
    True

    Changing the class count only moves the single final linear layer:

    >>> geron_googlenet(10)["total_params"] - r["total_params"]
    -1014750

    Padding keeps the stem alive even on tiny inputs, so the feature map
    bottoms out at 1x1 rather than vanishing:

    >>> geron_googlenet(10, input_size=16)["final_feature_map"]
    (1024, 1, 1)

    References
    ----------
    Géron Ch 12
    """
    C = int(n_classes)
    if C < 1:
        raise ValueError(f"geron_googlenet: n_classes must be >= 1, got {n_classes!r}")
    S = int(input_size)
    ch = int(in_channels)
    if S < 1 or ch < 1:
        raise ValueError("geron_googlenet: input_size and in_channels must be >= 1")
    p_drop = float(dropout)
    if not (0.0 <= p_drop < 1.0):
        raise ValueError(f"geron_googlenet: dropout must lie in [0, 1), got {p_drop}")

    layers = []
    size = S

    def conv(name, k, s, pad, out_ch):
        nonlocal size, ch
        o = (size - k + 2 * pad) // s + 1
        if o < 1:
            raise ValueError(
                f"geron_googlenet: input_size={S} is too small; a conv layer would produce a {o}x{o} feature map"
            )
        params = out_ch * (k * k * ch) + out_ch
        layers.append({"kind": "conv", "name": name, "out": o, "channels": out_ch, "params": params})
        size, ch = o, out_ch

    def pool(name, k=3, s=2, pad=1):
        nonlocal size
        o = (size - k + 2 * pad) // s + 1
        if o < 1:
            raise ValueError(
                f"geron_googlenet: input_size={S} is too small; a pool layer would produce a {o}x{o} feature map"
            )
        layers.append({"kind": "pool", "name": name, "out": o, "channels": ch, "params": 0})
        size = o

    conv("conv1", 7, 2, 3, 64)
    pool("pool1")
    conv("conv2", 1, 1, 0, 64)
    conv("conv3", 3, 1, 1, 192)
    pool("pool2")

    modules = []
    for name, o1, r3, o3, r5, o5, pp in _INCEPTION:
        mod = inception_module(ch, o1, r3, o3, r5, o5, pp)
        mod["name"] = name
        mod["in_channels"] = int(ch)
        mod["out"] = int(size)
        modules.append(mod)
        layers.append({"kind": "inception", "name": name, "out": int(size), "channels": mod["out_channels"], "params": mod["params"]})
        ch = mod["out_channels"]
        if name in _POOL_AFTER:
            pool(f"pool_{name}")

    fc_params = ch * C + C
    layers.append({"kind": "gap", "name": "global_avg_pool", "out": 1, "channels": ch, "params": 0})
    layers.append({"kind": "fc", "name": "classifier", "out": C, "channels": C, "params": fc_params, "dropout": p_drop})

    total = int(sum(l["params"] for l in layers))
    saving = int(sum(m["reduction_saving"] for m in modules))

    return RichResult(
        title="GoogLeNet / Inception",
        summary_lines=[("Total parameters", total), ("Inception modules", len(modules)), ("Classes", C)],
        tables=[{
            "title": "Inception modules",
            "headers": ["name", "in", "out", "params"],
            "rows": [[m["name"], m["in_channels"], m["out_channels"], m["params"]] for m in modules],
        }],
        interpretation="Branch outputs are concatenated, so widths add; the 1x1 reductions are what keep that affordable.",
        payload={
            "layers": layers,
            "modules": modules,
            "total_params": total,
            "conv_params": int(sum(l["params"] for l in layers if l["kind"] == "conv")),
            "inception_params": int(sum(l["params"] for l in layers if l["kind"] == "inception")),
            "fc_params": int(fc_params),
            "output_shape": (C,),
            "final_feature_map": (int(ch), int(size), int(size)),
            "total_reduction_saving": saving,
            "dropout": p_drop,
            "estimate": float(total),
            "n": int(len(layers)),
            "method": "GoogLeNet architecture resolved to concrete shapes and exact per-branch parameter counts",
        },
    )


def cheatsheet():
    return "hmgoog: GoogLeNet/Inception with parallel filter modules"
