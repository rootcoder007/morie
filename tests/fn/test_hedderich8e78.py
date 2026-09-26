"""hedderich8e78 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e78 import hedderich_chapter_8_equation_78


def test_hedderich8e78_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_78(x=None)
