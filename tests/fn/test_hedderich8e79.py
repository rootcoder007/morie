"""hedderich8e79 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e79 import hedderich_chapter_8_equation_79


def test_hedderich8e79_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_79(x=None)
