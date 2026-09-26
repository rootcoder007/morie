"""hedderich9u1481 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1481 import hedderich_chapter_9_unnumbered_1481


def test_hedderich9u1481_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1481(x=None)
