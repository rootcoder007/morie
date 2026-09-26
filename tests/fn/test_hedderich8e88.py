"""hedderich8e88 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e88 import hedderich_chapter_8_equation_88


def test_hedderich8e88_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_88(x=None)
