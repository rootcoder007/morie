"""hedderich8e65 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e65 import hedderich_chapter_8_equation_65


def test_hedderich8e65_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_65(x=None)
