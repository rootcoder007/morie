"""hedderich9u298 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u298 import hedderich_chapter_9_unnumbered_298


def test_hedderich9u298_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_298(x=None)
