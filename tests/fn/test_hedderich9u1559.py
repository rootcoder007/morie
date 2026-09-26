"""hedderich9u1559 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1559 import hedderich_chapter_9_unnumbered_1559


def test_hedderich9u1559_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1559(x=None)
