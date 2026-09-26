"""hedderich9u943 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u943 import hedderich_chapter_9_unnumbered_943


def test_hedderich9u943_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_943(x=None)
