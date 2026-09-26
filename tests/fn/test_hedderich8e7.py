"""hedderich8e7 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e7 import hedderich_chapter_8_equation_7


def test_hedderich8e7_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_7(x=None)
