"""hedderich2e48 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich2e48 import hedderich_chapter_2_equation_48


def test_hedderich2e48_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_2_equation_48(x=None)
