"""hedderich9u184 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u184 import hedderich_chapter_9_unnumbered_184


def test_hedderich9u184_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_184(x=None)
