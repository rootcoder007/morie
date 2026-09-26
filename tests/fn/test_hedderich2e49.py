"""hedderich2e49 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich2e49 import hedderich_chapter_2_equation_49


def test_hedderich2e49_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_2_equation_49(x=None)
