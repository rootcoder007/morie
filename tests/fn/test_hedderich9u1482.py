"""hedderich9u1482 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1482 import hedderich_chapter_9_unnumbered_1482


def test_hedderich9u1482_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1482(x=None)
