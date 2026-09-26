"""hedderich9u1485 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1485 import hedderich_chapter_9_unnumbered_1485


def test_hedderich9u1485_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1485(x=None)
