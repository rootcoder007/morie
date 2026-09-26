"""hedderich9u296 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u296 import hedderich_chapter_9_unnumbered_296


def test_hedderich9u296_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_296(x=None)
