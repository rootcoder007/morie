"""hedderich8e50 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e50 import hedderich_chapter_8_equation_50


def test_hedderich8e50_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_50(x=None)
