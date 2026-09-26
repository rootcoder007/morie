"""hedderich7e40 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich7e40 import hedderich_chapter_7_equation_40


def test_hedderich7e40_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_7_equation_40(x=None)
