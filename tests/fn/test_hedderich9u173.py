"""hedderich9u173 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u173 import hedderich_chapter_9_unnumbered_173


def test_hedderich9u173_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_173(x=None)
