"""hedderich9u2914 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u2914 import hedderich_chapter_9_unnumbered_2914


def test_hedderich9u2914_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_2914(x=None)
