"""hedderich5e85 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich5e85 import hedderich_chapter_5_equation_85


def test_hedderich5e85_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_5_equation_85(x=None)
