"""hedderich2e46 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich2e46 import hedderich_chapter_2_equation_46


def test_hedderich2e46_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_2_equation_46(x=None)
