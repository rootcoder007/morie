"""hedderich8e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e9 import hedderich_chapter_8_equation_9


def test_hedderich8e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_9(x=None)
