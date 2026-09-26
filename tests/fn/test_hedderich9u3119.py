"""hedderich9u3119 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u3119 import hedderich_chapter_9_unnumbered_3119


def test_hedderich9u3119_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_3119(x=None)
