"""hedderich8e27 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e27 import hedderich_chapter_8_equation_27


def test_hedderich8e27_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_27(x=None)
