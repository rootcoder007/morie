"""hedderich9u188 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u188 import hedderich_chapter_9_unnumbered_188


def test_hedderich9u188_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_188(x=None)
