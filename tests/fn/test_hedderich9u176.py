"""hedderich9u176 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u176 import hedderich_chapter_9_unnumbered_176


def test_hedderich9u176_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_176(x=None)
