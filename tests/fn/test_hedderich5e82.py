"""hedderich5e82 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich5e82 import hedderich_chapter_5_equation_82


def test_hedderich5e82_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_5_equation_82(x=None)
