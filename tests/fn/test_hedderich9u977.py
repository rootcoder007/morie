"""hedderich9u977 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u977 import hedderich_chapter_9_unnumbered_977


def test_hedderich9u977_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_977(x=None)
