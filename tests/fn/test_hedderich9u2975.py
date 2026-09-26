"""hedderich9u2975 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2975 import hedderich_chapter_9_unnumbered_2975


def test_hedderich9u2975_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2975(x=None)
