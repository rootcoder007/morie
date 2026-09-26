"""hedderich8e44 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e44 import hedderich_chapter_8_equation_44


def test_hedderich8e44_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_44(x=None)
