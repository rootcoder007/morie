"""hedderich8e4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e4 import hedderich_chapter_8_equation_4


def test_hedderich8e4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_4(x=None)
