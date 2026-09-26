"""hedderich9u1686 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich9u1686 import hedderich_chapter_9_unnumbered_1686


def test_hedderich9u1686_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_9_unnumbered_1686(x=None)
