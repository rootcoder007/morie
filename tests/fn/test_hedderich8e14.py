"""hedderich8e14 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e14 import hedderich_chapter_8_equation_14


def test_hedderich8e14_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_14(x=None)
