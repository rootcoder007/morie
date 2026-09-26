"""hedderich6e6 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich6e6 import hedderich_chapter_6_equation_6


def test_hedderich6e6_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_6_equation_6(x=None)
