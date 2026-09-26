"""hedderich9u1401 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1401 import hedderich_chapter_9_unnumbered_1401


def test_hedderich9u1401_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1401(x=None)
