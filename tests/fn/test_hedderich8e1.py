"""hedderich8e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e1 import hedderich_chapter_8_equation_1


def test_hedderich8e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_1(x=None)
