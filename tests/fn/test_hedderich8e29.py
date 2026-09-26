"""hedderich8e29 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e29 import hedderich_chapter_8_equation_29


def test_hedderich8e29_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_29(x=None)
