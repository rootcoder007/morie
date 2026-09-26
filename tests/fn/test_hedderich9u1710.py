"""hedderich9u1710 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1710 import hedderich_chapter_9_unnumbered_1710


def test_hedderich9u1710_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1710(x=None)
