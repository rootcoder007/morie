"""hedderich8e16 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e16 import hedderich_chapter_8_equation_16


def test_hedderich8e16_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_16(x=None)
