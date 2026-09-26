"""hedderich7e54 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich7e54 import hedderich_chapter_7_equation_54


def test_hedderich7e54_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_7_equation_54(x=None)
