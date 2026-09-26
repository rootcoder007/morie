"""hedderich8e2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e2 import hedderich_chapter_8_equation_2


def test_hedderich8e2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_2(x=None)
