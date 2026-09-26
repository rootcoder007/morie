"""hedderich9u340 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u340 import hedderich_chapter_9_unnumbered_340


def test_hedderich9u340_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_340(x=None)
