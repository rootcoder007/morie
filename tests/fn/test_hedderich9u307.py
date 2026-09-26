"""hedderich9u307 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u307 import hedderich_chapter_9_unnumbered_307


def test_hedderich9u307_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_307(x=None)
