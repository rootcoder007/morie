"""hedderich9u308 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u308 import hedderich_chapter_9_unnumbered_308


def test_hedderich9u308_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_308(x=None)
