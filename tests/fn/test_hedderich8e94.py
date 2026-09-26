"""hedderich8e94 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e94 import hedderich_chapter_8_equation_94


def test_hedderich8e94_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_94(x=None)
