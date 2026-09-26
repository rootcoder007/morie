"""hedderich9u953 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u953 import hedderich_chapter_9_unnumbered_953


def test_hedderich9u953_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_953(x=None)
