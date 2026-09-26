"""hedderich9u148 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u148 import hedderich_chapter_9_unnumbered_148


def test_hedderich9u148_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_148(x=None)
