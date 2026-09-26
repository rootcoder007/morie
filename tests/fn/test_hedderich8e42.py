"""hedderich8e42 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e42 import hedderich_chapter_8_equation_42


def test_hedderich8e42_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_42(x=None)
