"""hedderich8e53 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e53 import hedderich_chapter_8_equation_53


def test_hedderich8e53_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_53(x=None)
