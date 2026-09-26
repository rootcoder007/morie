"""hedderich9u294 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u294 import hedderich_chapter_9_unnumbered_294


def test_hedderich9u294_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_294(x=None)
