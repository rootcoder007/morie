"""hedderich9u181 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u181 import hedderich_chapter_9_unnumbered_181


def test_hedderich9u181_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_181(x=None)
