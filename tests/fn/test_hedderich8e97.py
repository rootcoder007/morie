"""hedderich8e97 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e97 import hedderich_chapter_8_equation_97


def test_hedderich8e97_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_97(x=None)
