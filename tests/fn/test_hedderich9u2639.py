"""hedderich9u2639 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2639 import hedderich_chapter_9_unnumbered_2639


def test_hedderich9u2639_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2639(x=None)
