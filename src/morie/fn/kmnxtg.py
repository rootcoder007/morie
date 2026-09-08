# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NExT-GPT any-to-any: modality encoders -> LLM -> modality decoders."""

from ._richresult import RichResult

__all__ = ["kamath_nextgpt_any2any"]


def kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders,
                           output_modalities=None):
    """y_m = Decoder_m(LLM([Encoder_in(x_in)])).

    Orchestration only: every learned piece is the caller's, and the
    contract is enforced at each hop instead of being assumed. An
    input modality with no encoder, or a requested output modality
    with no decoder, is an error -- silently dropping a modality is
    how an "any-to-any" system quietly becomes text-to-text.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, NExT-GPT.

    Examples
    --------
    >>> enc = {"text": lambda s: [len(s)], "image": lambda a: [sum(a)]}
    >>> dec = {"audio": lambda h: ("wav", h)}
    >>> out = kamath_nextgpt_any2any(
    ...     {"text": "abc", "image": [1, 2]}, enc,
    ...     lambda feats: [feats["image"][0], feats["text"][0]], dec)
    >>> out["outputs"]["audio"]
    ('wav', [3, 3])
    >>> out["input_modalities"]
    ['image', 'text']
    """
    if not isinstance(inputs_by_modality, dict) or not inputs_by_modality:
        raise ValueError(
            "inputs_by_modality must be a non-empty dict "
            "{modality: input}.")
    if not isinstance(encoders, dict):
        raise ValueError("encoders must be a dict {modality: callable}.")
    if not isinstance(decoders, dict) or not decoders:
        raise ValueError(
            "decoders must be a non-empty dict {modality: callable}.")
    if not callable(llm):
        raise ValueError("llm must be callable {modality: features} -> state.")

    in_mods = sorted(inputs_by_modality)
    missing = [m for m in in_mods if m not in encoders]
    if missing:
        raise ValueError(
            f"no encoder for input modalities {missing}; an any-to-any "
            "model may not silently drop an input.")
    feats = {}
    for m in in_mods:
        f = encoders[m]
        if not callable(f):
            raise ValueError(f"the encoder for {m!r} is not callable.")
        e = f(inputs_by_modality[m])
        if e is None:
            raise ValueError(f"the encoder for {m!r} returned nothing.")
        feats[m] = e

    state = llm(feats)
    if state is None:
        raise ValueError("the LLM returned no state to decode.")

    out_mods = sorted(decoders) if output_modalities is None \
        else list(output_modalities)
    missing_d = [m for m in out_mods if m not in decoders]
    if missing_d:
        raise ValueError(f"no decoder for requested modalities {missing_d}.")
    outputs = {}
    for m in out_mods:
        g = decoders[m]
        if not callable(g):
            raise ValueError(f"the decoder for {m!r} is not callable.")
        outputs[m] = g(state)
    return RichResult(payload={
        "outputs": outputs, "features": feats, "llm_state": state,
        "input_modalities": in_mods, "output_modalities": out_mods,
        "estimate": len(outputs),
        "n": len(in_mods) + len(out_mods),
        "method": "NExT-GPT any-to-any encode / LLM / decode pipeline"})


def cheatsheet():
    return "kmnxtg: encoders -> llm -> decoders, every modality accounted for"
