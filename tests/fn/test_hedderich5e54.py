"""hedderich5e54 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich5e54 import hedderich_chapter_5_equation_54


def test_hedderich5e54_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_5_equation_54(x=None)
