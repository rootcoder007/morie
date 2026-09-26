"""hedderich7e17 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich7e17 import hedderich_chapter_7_equation_17


def test_hedderich7e17_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_7_equation_17(x=None)
