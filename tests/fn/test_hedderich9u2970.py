"""hedderich9u2970 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2970 import hedderich_chapter_9_unnumbered_2970


def test_hedderich9u2970_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2970(x=None)
