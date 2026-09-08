# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.30: the self-diagnosis probability."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_self_diagnosis_prob"]


def _default_sdg(x, y):
    return f"{x}\nDoes the above text contain {y}?"


def kamath_ch6_self_diagnosis_prob(x, y, M, sdg=None):
    """p(y|x) = p_M("Yes" | sdg(x,y)) /
    sum_{w in {Yes, No}} p_M(w | sdg(x,y)).

    The model judges its OWN output: the two answer tokens are
    renormalised against each other, so all the probability mass the
    model spends on any other continuation is discarded -- that
    restriction is what makes the ratio a usable score. ``M`` is a
    callable prompt -> mapping containing "Yes" and "No"; ``sdg``
    defaults to the book's self-diagnosis template.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.30, printed
    p. 255.

    Examples
    --------
    >>> out = kamath_ch6_self_diagnosis_prob(
    ...     "I will hunt you down!", "a threat",
    ...     lambda prompt: {"Yes": 0.3, "No": 0.1})
    >>> round(out["estimate"], 12)
    0.75
    >>> "Does the above text contain a threat?" in out["prompt"]
    True
    """
    if not callable(M):
        raise ValueError("M must be a callable prompt -> "
                         "{'Yes': p, 'No': p}.")
    template = _default_sdg if sdg is None else sdg
    if not callable(template):
        raise ValueError("sdg must be a callable (x, y) -> prompt.")
    prompt = template(x, y)
    dist = M(prompt)
    if not isinstance(dist, dict):
        raise ValueError("M must return a mapping of token probabilities.")
    for w in ("Yes", "No"):
        if w not in dist:
            raise ValueError(f"M's output has no {w!r} probability.")
    py, pn = float(dist["Yes"]), float(dist["No"])
    if py < 0 or pn < 0 or py > 1 or pn > 1:
        raise ValueError("the Yes/No probabilities must lie in [0, 1].")
    tot = py + pn
    if tot <= 0:
        raise ValueError("the model puts no mass on either Yes or No; the "
                         "renormalised probability is undefined.")
    return RichResult(payload={
        "estimate": py / tot, "p_yes": py, "p_no": pn,
        "mass_on_yes_no": tot, "prompt": prompt, "attribute": y, "n": 2,
        "method": "self-diagnosis probability (Kamath Eq 6.30)"})


def cheatsheet():
    return "km106: p(Yes) renormalised against p(No) on the sdg prompt"
