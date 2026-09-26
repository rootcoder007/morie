"""hedderich9u178 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u178 import hedderich_chapter_9_unnumbered_178


def test_hedderich9u178_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_178(x=None)
