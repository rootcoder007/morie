"""hedderich9u182 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u182 import hedderich_chapter_9_unnumbered_182


def test_hedderich9u182_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_182(x=None)
