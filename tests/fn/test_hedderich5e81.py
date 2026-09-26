"""hedderich5e81 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich5e81 import hedderich_chapter_5_equation_81


def test_hedderich5e81_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_5_equation_81(x=None)
