"""hedderich8e60 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e60 import hedderich_chapter_8_equation_60


def test_hedderich8e60_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_60(x=None)
