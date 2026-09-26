"""hedderich8e98 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e98 import hedderich_chapter_8_equation_98


def test_hedderich8e98_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_98(x=None)
