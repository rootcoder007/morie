"""hedderich9u189 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u189 import hedderich_chapter_9_unnumbered_189


def test_hedderich9u189_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_189(x=None)
