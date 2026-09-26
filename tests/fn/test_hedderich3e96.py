"""hedderich3e96 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich3e96 import hedderich_chapter_3_equation_96


def test_hedderich3e96_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_3_equation_96(x=None)
