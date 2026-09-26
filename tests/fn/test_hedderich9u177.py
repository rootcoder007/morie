"""hedderich9u177 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u177 import hedderich_chapter_9_unnumbered_177


def test_hedderich9u177_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_177(x=None)
