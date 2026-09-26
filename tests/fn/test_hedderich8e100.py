"""hedderich8e100 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e100 import hedderich_chapter_8_equation_100


def test_hedderich8e100_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_100(x=None)
