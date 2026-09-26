"""hedderich7e31 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich7e31 import hedderich_chapter_7_equation_31


def test_hedderich7e31_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_7_equation_31(x=None)
