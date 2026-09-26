"""hedderich9u1469 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1469 import hedderich_chapter_9_unnumbered_1469


def test_hedderich9u1469_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1469(x=None)
