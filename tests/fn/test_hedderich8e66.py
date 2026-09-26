"""hedderich8e66 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e66 import hedderich_chapter_8_equation_66


def test_hedderich8e66_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_66(x=None)
