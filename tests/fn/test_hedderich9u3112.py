"""hedderich9u3112 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3112 import hedderich_chapter_9_unnumbered_3112


def test_hedderich9u3112_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3112(x=None)
