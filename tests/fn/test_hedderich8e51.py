"""hedderich8e51 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e51 import hedderich_chapter_8_equation_51


def test_hedderich8e51_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_51(x=None)
