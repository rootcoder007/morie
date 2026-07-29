# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.6: the translation prefix prompt template."""

import numpy as np

from ._richresult import RichResult
from .km046 import _fill_template, _result

__all__ = ["kamath_ch3_translate_prefix_prompt"]

TEMPLATE = "Translate the following English sentence to French: [x] [z]"


def kamath_ch3_translate_prefix_prompt(x, z=None, template=TEMPLATE):
    """x' = Translate the following English sentence to French: [x][z].

    Same prefix shape as Eq 3.5 with the task named explicitly, so the
    slot filling DELEGATES to km046's shared ``_fill_template``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.6, printed
    p. 100.

    Examples
    --------
    >>> kamath_ch3_translate_prefix_prompt("The cat sleeps.")["prompt"]
    'Translate the following English sentence to French: The cat sleeps. [z]'
    >>> kamath_ch3_translate_prefix_prompt("Hello.", "Bonjour.")["slot_filled"]
    True
    """
    prompt, filled = _fill_template(template, x, z)
    return _result(prompt, filled, "3.6", template)


def cheatsheet():
    return "km047: translation prefix template, shared km046 filler"
