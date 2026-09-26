"""hedderich6e25 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich6e25 import hedderich_chapter_6_equation_25


def test_hedderich6e25_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_6_equation_25(x=None)
