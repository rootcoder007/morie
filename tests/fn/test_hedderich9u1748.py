"""hedderich9u1748 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1748 import hedderich_chapter_9_unnumbered_1748


def test_hedderich9u1748_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1748(x=None)
