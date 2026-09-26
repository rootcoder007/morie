"""hedderich9u1470 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1470 import hedderich_chapter_9_unnumbered_1470


def test_hedderich9u1470_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1470(x=None)
