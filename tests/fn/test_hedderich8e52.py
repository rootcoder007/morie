"""hedderich8e52 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e52 import hedderich_chapter_8_equation_52


def test_hedderich8e52_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_52(x=None)
