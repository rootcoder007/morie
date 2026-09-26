"""hedderich2e47 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich2e47 import hedderich_chapter_2_equation_47


def test_hedderich2e47_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_2_equation_47(x=None)
