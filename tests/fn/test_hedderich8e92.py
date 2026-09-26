"""hedderich8e92 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e92 import hedderich_chapter_8_equation_92


def test_hedderich8e92_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_92(x=None)
