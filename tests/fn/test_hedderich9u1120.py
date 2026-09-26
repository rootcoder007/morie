"""hedderich9u1120 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1120 import hedderich_chapter_9_unnumbered_1120


def test_hedderich9u1120_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1120(x=None)
