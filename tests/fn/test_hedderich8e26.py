"""hedderich8e26 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e26 import hedderich_chapter_8_equation_26


def test_hedderich8e26_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_26(x=None)
