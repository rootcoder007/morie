"""hedderich8e13 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e13 import hedderich_chapter_8_equation_13


def test_hedderich8e13_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_13(x=None)
