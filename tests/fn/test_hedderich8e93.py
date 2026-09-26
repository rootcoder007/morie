"""hedderich8e93 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e93 import hedderich_chapter_8_equation_93


def test_hedderich8e93_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_93(x=None)
