"""hedderich9u183 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u183 import hedderich_chapter_9_unnumbered_183


def test_hedderich9u183_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_183(x=None)
