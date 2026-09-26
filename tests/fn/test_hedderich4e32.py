"""hedderich4e32 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich4e32 import hedderich_chapter_4_equation_32


def test_hedderich4e32_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_4_equation_32(x=None)
