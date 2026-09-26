"""hedderich8e19 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e19 import hedderich_chapter_8_equation_19


def test_hedderich8e19_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_19(x=None)
