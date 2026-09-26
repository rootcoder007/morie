"""hedderich9u1226 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1226 import hedderich_chapter_9_unnumbered_1226


def test_hedderich9u1226_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1226(x=None)
