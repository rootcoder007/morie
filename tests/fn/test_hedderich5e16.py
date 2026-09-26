"""hedderich5e16 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich5e16 import hedderich_chapter_5_equation_16


def test_hedderich5e16_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_5_equation_16(x=None)
