"""hedderich9u3163 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3163 import hedderich_chapter_9_unnumbered_3163


def test_hedderich9u3163_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3163(x=None)
