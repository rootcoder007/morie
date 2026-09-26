"""hedderich8e48 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e48 import hedderich_chapter_8_equation_48


def test_hedderich8e48_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_48(x=None)
