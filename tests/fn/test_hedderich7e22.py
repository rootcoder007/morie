"""hedderich7e22 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich7e22 import hedderich_chapter_7_equation_22


def test_hedderich7e22_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_7_equation_22(x=None)
